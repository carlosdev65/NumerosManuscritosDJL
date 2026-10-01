package com.example.mnist;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;
import javax.imageio.ImageIO;

/**
 * Baixa os arquivos binários originais do MNIST (formato IDX) diretamente da
 * internet — sem depender do módulo ai.djl:basicdataset — e grava cada
 * imagem individualmente como um arquivo .png real em disco, organizado em
 * uma subpasta por dígito (0 a 9):
 *
 * <pre>
 * images-training/
 *   0/00000.png, 0/00001.png, ...
 *   1/00000.png, ...
 *   ...
 * images-validation/
 *   0/00000.png, ...
 * </pre>
 *
 * Esse é o padrão "pasta por classe", usado por praticamente qualquer
 * framework de deep learning para carregar imagens direto do disco.
 * O TrainMnist lê essas pastas através da classe ImageFolderDataset.
 *
 * <p>Execute uma única vez (leva alguns minutos e grava ~70.000 arquivos):
 * <pre>
 * mvn exec:java -Dexec.mainClass="com.example.mnist.PrepareTrainingImages"
 * </pre>
 */
public final class PrepareTrainingImages {

    // Mirror confiável dos arquivos originais do MNIST. O site oficial de
    // Yann LeCun (yann.lecun.com) costuma falhar por certificado SSL
    // expirado, então usamos este espelho público mantido pela comunidade.
    private static final String MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/";

    // "Números mágicos" que identificam o tipo de cada arquivo IDX (formato
    // binário do MNIST): 2051 = arquivo de imagens, 2049 = arquivo de rótulos.
    private static final int IDX_IMAGES_MAGIC = 2051;
    private static final int IDX_LABELS_MAGIC = 2049;

    private PrepareTrainingImages() {}

    public static void main(String[] args) throws IOException {
        // Conjunto de treino: 60.000 imagens, usadas para ensinar a rede.
        exportSet(
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                Paths.get("images-training"),
                "treino");

        // Conjunto de validação (o "t10k" do MNIST): 10.000 imagens que a rede
        // nunca vê durante o treino, usadas para medir a acurácia real.
        exportSet(
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
                Paths.get("images-validation"),
                "validação");

        System.out.println("\nConcluído! Pastas geradas:");
        System.out.println("  - images-training/   (usada por TrainMnist para treinar)");
        System.out.println("  - images-validation/  (usada por TrainMnist para validar e por InferMnist para testar)");
    }

    /**
     * Baixa um par (imagens + rótulos) do MNIST e grava cada imagem como um
     * PNG dentro de {@code outputDir/<dígito>/}.
     */
    private static void exportSet(
            String imagesFileName, String labelsFileName, Path outputDir, String descricao)
            throws IOException {

        System.out.println("Baixando conjunto de " + descricao + "...");
        // Os arquivos vêm compactados (.gz); ficam em memória (são pequenos:
        // poucos MB) e são descompactados em seguida, em streaming.
        byte[] imagesGz = download(MIRROR + imagesFileName);
        byte[] labelsGz = download(MIRROR + labelsFileName);

        try (DataInputStream imagesIn =
                        new DataInputStream(new GZIPInputStream(new ByteArrayInputStream(imagesGz)));
                DataInputStream labelsIn =
                        new DataInputStream(new GZIPInputStream(new ByteArrayInputStream(labelsGz)))) {

            // Cabeçalho do arquivo IDX de imagens: 4 inteiros big-endian
            // (número mágico, quantidade de imagens, linhas, colunas).
            int imagesMagic = imagesIn.readInt();
            int numImages = imagesIn.readInt();
            int rows = imagesIn.readInt();
            int cols = imagesIn.readInt();

            // Cabeçalho do arquivo IDX de rótulos: número mágico + quantidade.
            int labelsMagic = labelsIn.readInt();
            int numLabels = labelsIn.readInt();

            if (imagesMagic != IDX_IMAGES_MAGIC || labelsMagic != IDX_LABELS_MAGIC) {
                throw new IOException("Formato IDX inesperado ao ler os arquivos do MNIST.");
            }
            if (numImages != numLabels) {
                throw new IOException("Quantidade de imagens e de rótulos não bate.");
            }

            System.out.println(numImages + " imagens de " + descricao + " encontradas. Gravando PNGs...");

            // Buffer reaproveitado: 28 x 28 = 784 bytes, um por pixel (0 = preto,
            // 255 = branco). As imagens do MNIST são traço branco em fundo preto.
            byte[] pixels = new byte[rows * cols];
            for (int i = 0; i < numImages; i++) {
                // As imagens e os rótulos estão na MESMA ordem nos dois
                // arquivos: a i-ésima imagem corresponde ao i-ésimo rótulo.
                imagesIn.readFully(pixels);
                int digit = labelsIn.readUnsignedByte();

                // A pasta com o nome do dígito ("7", "3"...) será o rótulo lido
                // depois pelo ImageFolderDataset. Criada sob demanda.
                Path classDir = outputDir.resolve(String.valueOf(digit));
                Files.createDirectories(classDir);

                // Monta uma imagem em escala de cinza (1 byte por pixel)
                // diretamente a partir dos bytes crus do MNIST.
                BufferedImage image = new BufferedImage(cols, rows, BufferedImage.TYPE_BYTE_GRAY);
                image.getRaster().setDataElements(0, 0, cols, rows, pixels);

                // Nome com zeros à esquerda (00000.png, 00001.png, ...); o número
                // é o índice da imagem no arquivo original do MNIST.
                Path file = classDir.resolve(String.format("%05d.png", i));
                ImageIO.write(image, "png", file.toFile());

                // Mostra o progresso a cada 10.000 imagens.
                if (i % 10_000 == 0) {
                    System.out.println("  " + i + " / " + numImages);
                }
            }
        }

        System.out.println("Pasta pronta: " + outputDir.toAbsolutePath());
    }

    /**
     * Baixa o conteúdo de uma URL inteira para a memória, usando o
     * HttpClient nativo do Java 11+ (sem bibliotecas extras).
     */
    private static byte[] download(String url) throws IOException {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder(URI.create(url)).build();
        try {
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new IOException(
                        "Falha ao baixar " + url + " (HTTP " + response.statusCode() + ")");
            }
            return response.body();
        } catch (InterruptedException e) {
            // Boa prática: restaura o sinal de interrupção da thread antes de
            // converter o erro em IOException.
            Thread.currentThread().interrupt();
            throw new IOException("Download interrompido: " + url, e);
        }
    }
}
