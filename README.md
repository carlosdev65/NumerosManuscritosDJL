# Reconhecimento de Dígitos Manuscritos com DJL + JavaFX

Projeto didático de ponta a ponta de **Deep Learning em Java**: prepara o
dataset MNIST como imagens PNG, **treina** uma rede neural, e permite
**classificar dígitos** tanto pela **linha de comando** quanto por uma
**interface gráfica desktop (JavaFX)**.

Construído com [Deep Java Library (DJL) 0.36.0](https://djl.ai/) (engine
PyTorch), Java 21 e Maven.

> O projeto **não** usa o módulo `ai.djl:basicdataset`. Os dados viram
> arquivos `.png` reais em disco e são lidos por um `Dataset` customizado
> (`ImageFolderDataset`) — o mesmo padrão "uma pasta por classe" que você
> usaria com suas próprias imagens.

---

## Sumário

1. [Visão geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Como executar](#como-executar)
5. [Como funciona](#como-funciona)
6. [Explicação das classes](#explicação-das-classes)
7. [Configuração (`pom.xml`)](#configuração-pomxml)
8. [Dicas e limitações](#dicas-e-limitações)
9. [Solução de problemas](#solução-de-problemas)
10. [Ideias para evoluir](#ideias-para-evoluir)

---

## Visão geral

```
 PrepareTrainingImages      TrainMnist                 InferMnist (CLI)
 baixa o MNIST e grava  →   treina a rede e salva  →   ou InferenceApp (JavaFX)
 PNGs em pastas             build/mnist-model/         classificam uma imagem
```

| Item | Valor |
|---|---|
| Tarefa | Classificação de imagens em 10 classes (dígitos 0–9) |
| Dataset | MNIST — 60.000 imagens de treino + 10.000 de validação (28×28, tons de cinza) |
| Modelo | MLP (rede densa): `784 → 128 → 64 → 10` |
| Treino | 5 épocas, lotes de 32, perda *softmax cross-entropy* |
| Engine | PyTorch (via DJL) |

---

## Pré-requisitos

- **JDK 21** (`java -version` deve mostrar 21.x)
- **Maven 3.8+** (`mvn -version`)
- **Internet** na primeira execução, para baixar o MNIST e os binários
  nativos do PyTorch e do JavaFX.

---

## Estrutura do projeto

```
djl-mnist-example/
├── pom.xml                          # dependências, profiles por SO e empacotamento (fat JAR)
├── README.md
├── images-training/                 # 60.000 PNGs de treino   (uma subpasta por dígito: 0/ … 9/)
├── images-validation/               # 10.000 PNGs de validação (idem)
├── build/mnist-model/               # modelo treinado (mnist-mlp-0005.params)
└── src/main/java/com/example/mnist/
    ├── PrepareTrainingImages.java   # baixa o MNIST cru e grava os PNGs
    ├── ImageFolderDataset.java      # Dataset customizado que lê as pastas de imagens
    ├── DigitClassifier.java         # arquitetura + carregamento do modelo + inferência
    ├── TrainMnist.java              # treina e salva o modelo
    ├── InferMnist.java              # inferência por linha de comando
    └── ui/
        ├── Launcher.java            # Main-Class real do JAR
        └── InferenceApp.java        # interface gráfica JavaFX
```

---

## Como executar

Execute os comandos **sempre na raiz do projeto** (a pasta do `pom.xml`): os
caminhos `images-training/`, `images-validation/` e `build/mnist-model/` são
relativos ao diretório atual.

### 1. Compilar

```bash
mvn clean compile
```

### 2. Gerar as imagens (uma única vez)

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.PrepareTrainingImages"
```

Baixa o MNIST original e grava `images-training/<dígito>/*.png` (60.000) e
`images-validation/<dígito>/*.png` (10.000). Leva alguns minutos.

> Se as pastas de imagens já existem (como neste projeto), **pule este passo**.

### 3. Treinar o modelo

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.TrainMnist"
```

O console mostra a perda e a acurácia de cada época. Ao final, o modelo é
salvo em `build/mnist-model/`. Com uma MLP simples, espera-se cerca de
**97–98% de acurácia** de validação (o valor exato varia a cada execução).

> Se `build/mnist-model/` já contém o modelo, também pode pular este passo.

### 4. Classificar um dígito

**a) Linha de comando**

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.InferMnist" \
    -Dexec.args="images-validation/7/00064.png"
```

Saída: as probabilidades de cada dígito e o dígito reconhecido.

**b) Interface gráfica (JavaFX)**

```bash
mvn clean package
java -jar target/djl-mnist-example-1.0.0.jar
```

Na janela:

1. Clique em **Selecionar imagem...** e escolha um `.png`/`.jpg`/`.bmp`
   (por exemplo, qualquer arquivo de `images-validation/<dígito>/`).
2. A imagem aparece na tela.
3. Clique em **Classificar dígito**: o dígito reconhecido aparece em
   destaque, com uma barra de probabilidade para cada classe (0–9).

---

## Como funciona

1. **Preparo dos dados** — `PrepareTrainingImages` lê os arquivos binários
   do MNIST (formato IDX) e grava cada imagem como PNG em
   `images-*/<dígito>/`. O **nome da pasta é o rótulo**.
2. **Leitura no treino** — `ImageFolderDataset` indexa os caminhos dos
   arquivos e só decodifica cada imagem quando o treino a pede (carregamento
   preguiçoso), convertendo-a em tensor `(1, 28, 28)` com valores entre 0 e 1.
3. **Treino** — `TrainMnist` ajusta os pesos da MLP em lotes de 32 imagens,
   por 5 épocas, medindo a acurácia no conjunto de validação.
4. **Inferência** — `DigitClassifier` recria **a mesma arquitetura**, carrega
   os pesos salvos e, para cada imagem: converte para cinza → redimensiona
   para 28×28 → normaliza → roda a rede → aplica `softmax` (probabilidades).

O pré-processamento do treino e o da inferência são equivalentes; isso é
essencial para o modelo funcionar bem fora do treino.

---

## Explicação das classes

### `PrepareTrainingImages`
Baixa os 4 arquivos `.gz` do MNIST de um espelho público, valida o
cabeçalho IDX (números mágicos 2051/2049) e grava as imagens como PNG. O
site oficial (`yann.lecun.com`) costuma ter problemas de certificado, por
isso usa-se o espelho `https://ossci-datasets.s3.amazonaws.com/mnist/`
(constante `MIRROR`, fácil de trocar se ficar indisponível).

### `ImageFolderDataset`
Subclasse de `RandomAccessDataset` do DJL, com o padrão *builder*:

```java
ImageFolderDataset ds = ImageFolderDataset.builder()
        .setRoot(Paths.get("images-training"))
        .setSampling(32, true)       // tamanho do lote + embaralhar
        .build();
ds.prepare(new ProgressBar());
```

| Método | Função |
|---|---|
| `prepare()` | Lista os `.png` de cada subpasta numérica e guarda (caminho, rótulo). Ignora pastas com nome não numérico e ordena os arquivos para manter a ordem determinística. |
| `get(index)` | Lê **uma** imagem do disco e devolve o par (tensor, rótulo). |
| `availableSize()` | Total de imagens indexadas. |

Para usar com **outro dataset seu**, basta a mesma estrutura de pastas
numeradas (`0/`, `1/`, …).

### `DigitClassifier`
"Fonte única de verdade" do modelo — usada por treino, CLI e interface:

| Membro | Função |
|---|---|
| `newBlock()` | Define a arquitetura (MLP 784→128→64→10). Compartilhada com o treino para nunca divergirem. |
| Construtor | Cria o modelo, aplica `newBlock()` e carrega os pesos de `build/mnist-model`. |
| `classify(Path)` | Lê a imagem e devolve as probabilidades (`Classifications`). |
| `Translator` interno | Pré-processa a imagem (cinza, 28×28, normalização) e pós-processa a saída (`softmax`). |
| `close()` | Libera memória nativa (`AutoCloseable` → use *try-with-resources*). |

### `TrainMnist`
Monta os datasets, cria o modelo, configura o treino (perda
`softmaxCrossEntropyLoss` + métrica `Accuracy`), executa
`EasyTrain.fit(...)` e salva o resultado. Hiperparâmetros (`BATCH_SIZE`,
`EPOCHS`) ficam como constantes no topo da classe.

### `InferMnist`
CLI mínima: lê o caminho da imagem, chama `DigitClassifier` e imprime o
resultado.

### `ui/InferenceApp`
Aplicação JavaFX:

| Método | Função |
|---|---|
| `start(Stage)` | Carrega o modelo uma vez (mostra alerta claro se ainda não foi treinado) e monta a janela. |
| `onSelectImage()` | Abre o seletor de arquivos nativo e exibe a imagem escolhida. |
| `onClassify()` | Roda a inferência numa `Task` (outra thread), para a janela **não travar**. |
| `showResults()` | Mostra o dígito em destaque + barras de probabilidade 0–9. |

### `ui/Launcher`
É o `Main-Class` do JAR. Existe por causa de uma particularidade do JavaFX:
se o `Main-Class` estende `javafx.application.Application`, o
`java -jar` falha com *"JavaFX runtime components are missing"* em um fat
JAR (classpath simples). O `Launcher` **não** estende `Application` e só
delega para `InferenceApp.main(args)`, contornando o problema.

---

## Configuração (`pom.xml`)

| Elemento | Por quê |
|---|---|
| BOM do DJL (`ai.djl:bom`) | Fixa em um só lugar versões compatíveis de todos os módulos DJL. |
| `api`, `model-zoo`, `pytorch-engine` | API do DJL, o bloco `Mlp` pronto e o engine que executa as operações (binários nativos baixados na 1ª execução). |
| `javafx-*` com `classifier ${javafx.platform}` | O JavaFX publica bibliotecas nativas por SO. |
| *Profiles* `windows` / `mac` / `linux` | Detectam o SO e ajustam `javafx.platform` automaticamente. |
| `exec-maven-plugin` | Permite `mvn exec:java -Dexec.mainClass=...`. |
| `maven-shade-plugin` | Gera o *fat JAR* (`java -jar`) na fase `package`. |
| `ServicesResourceTransformer` | Mescla os `META-INF/services`, sem o qual o DJL não acha o engine PyTorch dentro do JAR. |
| `ManifestResourceTransformer` | Define `ui.Launcher` como `Main-Class`. |

O arquivo `dependency-reduced-pom.xml` é gerado automaticamente pelo
shade-plugin e pode ser ignorado (está no `.gitignore`).

---

## Dicas e limitações

- **Fundo e traço:** o MNIST tem dígitos **brancos em fundo preto**. Uma foto
  de um número escuro em papel branco será lida "invertida" e provavelmente
  errada. Inverta as cores da imagem antes (ou treine com dados no mesmo
  estilo). Imagens de `images-validation/` funcionam direto.
- **Imagem centralizada:** o modelo espera um único dígito, centralizado e
  ocupando a maior parte da imagem, como no MNIST.
- **JAR por sistema operacional:** as bibliotecas nativas (JavaFX e PyTorch)
  variam por SO; o JAR só roda no sistema em que foi gerado. Gere um JAR em
  cada SO de destino.
- **Diretório de execução:** rode sempre da raiz do projeto (caminhos
  relativos).
- **Mudou a arquitetura?** Altere apenas `DigitClassifier.newBlock()` e
  **retreine**; pesos antigos não são compatíveis com outra estrutura.

---

## Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| Alerta "Não foi possível carregar o modelo treinado" | Rode `TrainMnist` antes (passo 3). |
| `Nenhuma subpasta de classe encontrada em ...` | Rode `PrepareTrainingImages` (passo 2) na raiz do projeto. |
| `Error: JavaFX runtime components are missing` | Execute pelo JAR gerado (`Launcher`), não por `InferenceApp` diretamente. |
| Erro de download do MNIST | Espelho fora do ar: troque a constante `MIRROR` em `PrepareTrainingImages`. |
| JAR não abre em outro computador | Sistema operacional diferente: gere o JAR naquele SO. |
| Resultado errado em foto própria | Veja "Fundo e traço" acima. |

---

## Ideias para evoluir

- Trocar a MLP por uma **CNN** (ex.: `ai.djl.basicmodelzoo.cv.classification.LeNet`) para acurácia acima de 99%.
- Adicionar um **canvas** na interface para desenhar o dígito com o mouse.
- Inverter automaticamente imagens de fundo claro antes da classificação.
- Usar o engine **MXNet** (mais leve em CPU): troque `pytorch-engine` por `ai.djl.mxnet:mxnet-engine` no `pom.xml`; o código Java não muda.
