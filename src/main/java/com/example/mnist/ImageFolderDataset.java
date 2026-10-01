package com.example.mnist;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.util.Progress;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Dataset customizado que lê imagens diretamente do disco, dispensando o
 * módulo {@code ai.djl:basicdataset}.
 *
 * <p>Espera uma pasta raiz organizada em uma subpasta por classe, onde o
 * NOME da subpasta é o próprio rótulo (padrão "pasta por classe", usado por
 * praticamente todo framework de deep learning):
 *
 * <pre>
 * images-training/
 *   0/  -&gt; imagens do dígito 0
 *   1/  -&gt; imagens do dígito 1
 *   ...
 *   9/  -&gt; imagens do dígito 9
 * </pre>
 *
 * <p>Diferente do Mnist do basicdataset (que carrega tudo de uma vez em um
 * grande array na memória), este dataset é "preguiçoso": {@link #prepare}
 * apenas indexa os caminhos dos arquivos, e cada imagem só é de fato lida e
 * decodificada do disco quando {@link #get} é chamado para aquele índice —
 * normalmente uma vez por época, por imagem, durante o treino.
 *
 * <p>Por herdar de {@link RandomAccessDataset}, ganha de graça o
 * embaralhamento e a divisão em lotes (batches) — basta chamar
 * {@code setSampling(batchSize, shuffle)} no builder.
 */
public final class ImageFolderDataset extends RandomAccessDataset {

    /** Pasta raiz que contém uma subpasta por classe. */
    private final Path root;

    // Duas listas "paralelas": imagePaths[i] é o arquivo e imageLabels[i] é o
    // dígito (rótulo) correspondente a ele.
    private final List<Path> imagePaths = new ArrayList<>();
    private final List<Integer> imageLabels = new ArrayList<>();

    /** Evita indexar as pastas duas vezes se prepare() for chamado de novo. */
    private boolean prepared;

    private ImageFolderDataset(Builder builder) {
        super(builder);
        this.root = builder.root;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Escaneia a pasta raiz uma única vez: para cada subpasta (0-9), lista
     * os arquivos .png e guarda (caminho, rótulo) em memória. As imagens em
     * si NÃO são carregadas aqui — só os caminhos.
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        // Só considera subpastas cujo nome é um número (o rótulo). Isso ignora
        // pastas estranhas (ex.: ".git", "tmp") em vez de quebrar no parseInt.
        File[] classDirs = root.toFile().listFiles(
                file -> file.isDirectory() && file.getName().matches("\\d+"));
        if (classDirs == null || classDirs.length == 0) {
            throw new IOException(
                    "Nenhuma subpasta de classe encontrada em " + root.toAbsolutePath()
                            + ". Rode a classe PrepareTrainingImages primeiro.");
        }
        // Ordena numericamente (0, 1, 2, ..., 10) em vez de alfabeticamente.
        Arrays.sort(classDirs, Comparator.comparingInt(dir -> Integer.parseInt(dir.getName())));

        for (File classDir : classDirs) {
            // O nome da pasta é o rótulo (ex.: pasta "7" -> dígito 7).
            int label = Integer.parseInt(classDir.getName());

            File[] files = classDir.listFiles(
                    (dir, name) -> name.toLowerCase(Locale.ROOT).endsWith(".png"));
            if (files == null) {
                continue;
            }
            // listFiles() não garante ordem; ordenar deixa a indexação
            // determinística (mesma ordem em qualquer máquina/execução).
            Arrays.sort(files);

            for (File file : files) {
                imagePaths.add(file.toPath());
                imageLabels.add(label);
            }
        }

        prepared = true;

        if (progress != null) {
            progress.reset("Indexando imagens em " + root, imagePaths.size());
            progress.end();
        }
    }

    /**
     * Carrega e decodifica UMA imagem do disco (a de índice {@code index}) e a
     * transforma no par (dado, rótulo) que o treinador espera.
     */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        int i = Math.toIntExact(index);

        // Lê o arquivo PNG do disco (só neste momento — carregamento preguiçoso).
        Image image = ImageFactory.getInstance().fromFile(imagePaths.get(i));

        // Converte para escala de cinza (1 canal) e depois para um tensor
        // float com valores entre 0 e 1 (formato: canal, altura, largura) — o
        // mesmo tratamento aplicado na inferência por DigitClassifier,
        // garantindo consistência entre treino e inferência.
        NDArray array = image.toNDArray(manager, Image.Flag.GRAYSCALE);
        NDArray data = NDImageUtils.toTensor(array);

        // O rótulo é um escalar com o número do dígito (0 a 9).
        NDArray label = manager.create((float) imageLabels.get(i));

        // Record = um exemplo de treino: (entradas, rótulos).
        return new Record(new NDList(data), new NDList(label));
    }

    /** Quantidade total de imagens indexadas (válido depois de {@link #prepare}). */
    @Override
    public long availableSize() {
        return imagePaths.size();
    }

    /**
     * Builder no mesmo padrão usado pelos datasets prontos do DJL
     * (ex.: Mnist.Builder). Herda de {@code RandomAccessDataset.BaseBuilder},
     * que já traz opções como {@code setSampling(batchSize, shuffle)}.
     */
    public static final class Builder extends BaseBuilder<Builder> {

        Path root;

        /** Define a pasta raiz (ex.: "images-training") a ser lida. */
        public Builder setRoot(Path root) {
            this.root = root;
            return this;
        }

        /** Exigido pelo padrão "builder genérico" do DJL: devolve o próprio builder. */
        @Override
        protected Builder self() {
            return this;
        }

        public ImageFolderDataset build() {
            return new ImageFolderDataset(this);
        }
    }
}
