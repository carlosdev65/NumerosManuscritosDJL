package com.example.mnist;

import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Versão de linha de comando da inferência: carrega o modelo treinado por
 * {@link TrainMnist} e classifica uma imagem de dígito manuscrito informada
 * como argumento.
 *
 * <p>Toda a lógica de carregar o modelo e traduzir imagem → tensor →
 * probabilidades está em {@link DigitClassifier}, reaproveitada também pela
 * interface gráfica ({@code com.example.mnist.ui.InferenceApp}). Esta classe
 * só lê o argumento e imprime o resultado.
 *
 * <p>Exemplo:
 * <pre>
 * mvn exec:java -Dexec.mainClass="com.example.mnist.InferMnist" \
 *     -Dexec.args="images-validation/7/00003.png"
 * </pre>
 */
public final class InferMnist {

    private InferMnist() {}

    public static void main(String[] args)
            throws IOException, MalformedModelException, TranslateException {

        // Sem argumento, mostra como usar e encerra.
        if (args.length < 1) {
            System.out.println("Uso: mvn exec:java -Dexec.mainClass=\"com.example.mnist.InferMnist\" "
                    + "-Dexec.args=\"caminho/para/imagem.png\"");
            return;
        }

        Path imagePath = Path.of(args[0]);

        // try-with-resources: garante que o modelo será fechado (liberando a
        // memória nativa) mesmo se ocorrer algum erro durante a inferência.
        try (DigitClassifier classifier = new DigitClassifier()) {
            Classifications classifications = classifier.classify(imagePath);

            System.out.println("Imagem: " + imagePath.toAbsolutePath());
            System.out.println("Resultado da predição (probabilidade por dígito):");
            System.out.println(classifications);

            // best() devolve a classe com a maior probabilidade.
            Classifications.Classification best = classifications.best();
            System.out.println("\n>>> Dígito reconhecido: " + best.getClassName()
                    + " (confiança " + String.format("%.2f%%", best.getProbability() * 100) + ")");
        }
    }
}
