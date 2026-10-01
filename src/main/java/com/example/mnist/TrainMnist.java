package com.example.mnist;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Treina uma rede neural simples (MLP) para reconhecer dígitos manuscritos.
 *
 * <p>Este exemplo NÃO usa o módulo {@code ai.djl:basicdataset} (e a classe
 * pronta Mnist). Em vez disso, lê as imagens diretamente das pastas
 * {@code images-training} e {@code images-validation}, geradas previamente por
 * {@link PrepareTrainingImages}, através do dataset customizado
 * {@link ImageFolderDataset}.
 *
 * <p>Fluxo: dataset → modelo → configuração do treino → treino → salvar.
 * A arquitetura da rede e as constantes (nome do modelo, pasta de saída)
 * vêm de {@link DigitClassifier}, o que garante que a inferência carregue o
 * modelo exatamente com a mesma estrutura usada aqui.
 */
public final class TrainMnist {

    /** Quantas imagens são processadas juntas a cada passo de atualização dos pesos. */
    private static final int BATCH_SIZE = 32;

    /** Quantas vezes o conjunto de treino completo é percorrido. */
    private static final int EPOCHS = 5;

    private TrainMnist() {}

    public static void main(String[] args) throws IOException, TranslateException {
        // Garante que a pasta de saída do modelo exista.
        Path modelDir = DigitClassifier.DEFAULT_MODEL_DIR;
        Files.createDirectories(modelDir);

        // 1) DATASETS
        // O ImageFolderDataset aponta para as pastas com os .png gerados por
        // PrepareTrainingImages. prepare() apenas lista os arquivos; cada
        // imagem só é lida do disco durante o treino, sob demanda.

        // Treino: lotes embaralhados (shuffle = true) a cada época, para que a
        // rede não "decore" a ordem das imagens.
        ImageFolderDataset trainingSet = ImageFolderDataset.builder()
                .setRoot(Paths.get("images-training"))
                .setSampling(BATCH_SIZE, true)
                .build();
        trainingSet.prepare(new ProgressBar());

        // Validação: imagens que a rede NÃO vê no treino, usadas só para medir
        // a acurácia real. Não precisa embaralhar (shuffle = false).
        ImageFolderDataset validationSet = ImageFolderDataset.builder()
                .setRoot(Paths.get("images-validation"))
                .setSampling(BATCH_SIZE, false)
                .build();
        validationSet.prepare(new ProgressBar());

        // 2) ARQUITETURA DO MODELO
        // A estrutura (MLP 784 -> 128 -> 64 -> 10) é definida em
        // DigitClassifier.newBlock(), compartilhada com a inferência.
        Model model = Model.newInstance(DigitClassifier.MODEL_NAME);
        model.setBlock(DigitClassifier.newBlock());

        // 3) CONFIGURAÇÃO DO TREINO
        // - Loss (perda): softmaxCrossEntropyLoss, a escolha padrão para
        //   classificação com várias classes. Mede o quão "errada" está a
        //   previsão; o treino tenta minimizar esse valor.
        // - Accuracy: métrica de acompanhamento (% de acertos).
        // - Logging: imprime o progresso de cada época no console.
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // 4) TREINAMENTO
        // O Trainer guarda recursos nativos, por isso é aberto em
        // try-with-resources (fechado automaticamente ao final).
        try (Trainer trainer = model.newTrainer(config)) {

            // Informa o formato de UMA entrada para o Trainer alocar os pesos:
            // (lote, canal, altura, largura) = (1, 1, 28, 28). Cada amostra do
            // ImageFolderDataset tem formato (1, 28, 28) — canal, altura,
            // largura — e o Mlp achata isso para 784 valores internamente.
            trainer.initialize(new Shape(1, 1, DigitClassifier.IMAGE_HEIGHT, DigitClassifier.IMAGE_WIDTH));

            trainer.setMetrics(new Metrics());

            // Loop de treino: para cada época, percorre os lotes do conjunto
            // de treino (ajustando os pesos) e depois avalia no de validação.
            EasyTrain.fit(trainer, EPOCHS, trainingSet, validationSet);

            TrainingResult result = trainer.getTrainingResult();
            System.out.println("Acurácia final de validação: "
                    + result.getValidateEvaluation("Accuracy"));
        }

        // 5) SALVAR O MODELO
        // Grava os pesos em build/mnist-model/mnist-mlp-0005.params (o 0005
        // indica a época). É esse arquivo que DigitClassifier carrega depois.
        model.save(modelDir, DigitClassifier.MODEL_NAME);
        System.out.println("Modelo salvo em: " + modelDir.toAbsolutePath());

        model.close();
    }
}
