# Proposta #1 — Tornar o acquire de blocos realmente assíncrono

## PROBLEMA

O `router_mm_acquire` atual executa a promoção RAM→VRAM de um bloco de forma
**síncrona**: dispara `cudaMemcpyAsync` e imediatamente chama
`cudaStreamSynchronize` antes de retornar.

```c
// router_memory.cu — caminho atual
cudaMemcpyAsync(vram[slot], ram[bloco], bytes, H2D, stream);
cudaStreamSynchronize(stream);   // <-- bloqueia a thread chamadora aqui
```

Isso anula o propósito do transfer assíncrono. O primitivo correto
(`router_mem_h2d_async`) já existe e não sincroniza, mas o caminho de
alto-nível (`router_mm_acquire`) — que é o que o path real consome — não o usa
de forma assíncrona. O resultado: a CPU só volta a computar depois que a cópia
H2D termina, portanto **não há overlap entre transferência e computação**,
que é exatamente o ganho que a arquitetura de streams foi criada para entregar.

## HIPÓTESE

Se o `acquire` retornar imediatamente após emitir a cópia (deixando o bloco em
um estado "loading") e a sincronização for adiada para o ponto em que o bloco
é realmente consumido (`wait_acquire`), então várias promoções de blocos
consecutivas (ex.: os experts do próximo layer) podem ser enviadas em rajada e
sobrepostas à computação do token anterior. Em um regime com alta taxa de
miss de cache (o cenário real de 8 GB RAM + 4 GB VRAM), isso deve reduzir o
tempo por token, porque a latência H2D deixa de ser serializada com o compute.

As seguintes premissas são empiricamente verdadeiras neste projeto:

1. Os blocos (experts) do próximo layer são **previsíveis** — o roteador e o
   "routing predictor bigram" já alcançaram 100% de precisão nos testes do
   usuário. Portanto, podemos emitir `acquire_async` para o próximo layer
   enquanto ainda computamos o layer atual.
2. `cudaMemcpyAsync` em stream non-blocking já é não-bloqueante para a CPU.

## IMPLEMENTAÇÃO

Adiciona a API assíncrona **ao lado** da síncrona, sem remover nada:

- `router_mm_acquire_async(m, block_id, bytes, *out_slot)` — reserva slot,
  dispara H2D, retorna imediatamente; bloco entra no estado `loading`.
- `router_mm_wait_acquire(m, block_id)` — sincroniza o stream do bloco e
  promove `loading → resident`.
- `router_mm_is_loading(m, block_id)` — consulta de estado.

Regras de segurança implementadas:

- **Sem transferência duplicada**: se o bloco já está `loading`, um segundo
  `acquire_async` retorna o slot já reservado em vez de emitir outra cópia.
- **Eviction não reutiliza slot em loading**: `find_free_vram` e a seleção de
  vítima LRU pulam qualquer slot cujo dono esteja em `loading`, até o
  `wait_acquire` correspondente ter sincronizado.
- **`unregister_block` sincroniza antes de liberar**: se o bloco tem H2D em
  voo, a unreg sincroniza primeiro para não liberar o RAM slot que a cópia
  ainda lê.
- **Caminho legado intacto**: `router_mm_acquire` continua síncrono e agora
  apenas detecta (e aguarda) um estado `loading` pré-existente para nunca
  duplicar transferência, mantendo sua semântica original.

## RISCO

- **Baixo.** Nenhum componente funcional é removido ou modificado de forma
  incompatível. O caminho síncrono legado mantém exatamente o mesmo
  comportamento. A mudança é puramente aditiva (novas funções + nova API).
- Risco residual: um consumidor que chame `acquire_async` e leia o VRAM
  pointer **sem** chamar `wait_acquire` leria dados não-inicializados. Esse
  risco é mitigado pela convenção documentada e pelo nome explícito
  (`wait_acquire`) — mas é uma disciplina de API, não uma garantia do tipo.
- Risco de build: requer recompilar `router_memory.cu` e re-testar o ABI
  (os símbolos novos são aditivos, então o ABI legado não muda).

## COMO VALIDAR

1. **Teste lógico (sem GPU)** — `simulate_async_acquire.py` em
   `experimental/async_memory/` replica a máquina de estados (loading →
   resident, skip de eviction, sem double-transfer) e verifica os invariantes
   com assertions. Executável em qualquer máquina.
2. **Teste real (no hardware do usuário)** — `native_memory/` smoke estendido:
   disparar `acquire_async` para N blocos consecutivos, medir o tempo de
   retorno acumulado (deve ser ~0, sem esperar cada H2D), então `wait_acquire`
   em cada um e conferir que todos promovem para resident com dados corretos.
3. **Benchmark de overlap** — comparar o tempo por token do caminho síncrono
   vs. o caminho "prefetch async do próximo layer" usando o mesmo seed de
   tokens. Espera-se redução mensurável quando houver miss de cache.

## CRITÉRIO DE ACEITAÇÃO

A mudança é considerada válida se **ambas** forem verdadeiras:

1. O smoke test assíncrono passa no hardware real (todos os blocos promovem
   para resident, dados corretos, sem transferência duplicada, sem corrupção
   de slot por eviction em loading).
2. O benchmark de overlap mostra redução **mensurável e reproduzível** no
   tempo por token em regime de miss de cache, comparado ao caminho síncrono.

Se o ganho não for mensurável (ex.: porque o gargalo real é dequant/compute e
não H2D), a proposta é **descartada conforme a regra do projeto** — não se
mantém complexidade sem ganho mensurável. Mesmo nesse caso, a API assíncrona
permanece disponível e documentada como base para a Proposta #3 (prefetch)
e #5 (router-aware prefetch), sem custo de manutenção.