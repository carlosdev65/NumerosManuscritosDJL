package com.example.mnist.ui;

import ai.djl.modality.Classifications;
import com.example.mnist.DigitClassifier;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Interface gráfica (JavaFX) para reconhecimento de dígitos manuscritos.
 *
 * Fluxo da aplicação:
 *   1) Ao abrir, carrega o modelo treinado (build/mnist-model), uma única vez.
 *   2) O usuário clica em "Selecionar imagem..." e escolhe um arquivo local
 *      (PNG/JPG/BMP) através de um FileChooser nativo do sistema operacional.
 *   3) A imagem escolhida é exibida na tela.
 *   4) Ao clicar em "Classificar dígito", a inferência roda em uma thread
 *      separada (para não travar a interface) e o resultado — o dígito
 *      reconhecido e a probabilidade de cada classe (0-9) — é exibido.
 */
public class InferenceApp extends Application {

    /** Modelo carregado (criado em start() e fechado ao fechar a janela). */
    private DigitClassifier classifier;

    /** Caminho da imagem escolhida pelo usuário; null enquanto nada foi escolhido. */
    private Path selectedImagePath;

    // Componentes de interface que precisam ser acessados por mais de um
    // método (por isso são campos, e não variáveis locais de start()).
    private final ImageView imageView = new ImageView();
    private final Label statusLabel = new Label("Selecione uma imagem para começar.");
    private final Button classifyButton = new Button("Classificar dígito");
    private final VBox resultsBox = new VBox(6);      // linhas com as probabilidades
    private final ProgressIndicator progressIndicator = new ProgressIndicator();

    /** Chamado pelo Launcher (ou diretamente, em ambiente de desenvolvimento). */
    public static void main(String[] args) {
        launch(args);
    }

    /**
     * Ponto de entrada do ciclo de vida do JavaFX: chamado na Application
     * Thread depois que o framework é inicializado. "stage" é a janela
     * principal.
     */
    @Override
    public void start(Stage stage) {
        // 1) CARREGAR O MODELO
        // Feito uma única vez, ao abrir a janela. Se o modelo ainda não
        // tiver sido treinado, mostramos um erro claro em vez de travar.
        try {
            classifier = new DigitClassifier();
        } catch (IOException e) {
            showFatalErrorAndExit(
                    "Não foi possível carregar o modelo treinado em build/mnist-model.\n\n"
                            + "Rode primeiro:\n"
                            + "  mvn exec:java -Dexec.mainClass=\"com.example.mnist.PrepareTrainingImages\"\n"
                            + "  mvn exec:java -Dexec.mainClass=\"com.example.mnist.TrainMnist\"\n\n"
                            + "Detalhe: " + e.getMessage());
            return;
        } catch (Exception e) {
            showFatalErrorAndExit("Erro inesperado ao carregar o modelo:\n" + e.getMessage());
            return;
        }

        // 2) MONTAR A INTERFACE
        // Barra de botões (topo): "Selecionar imagem..." e "Classificar dígito".
        Button selectButton = new Button("Selecionar imagem...");
        selectButton.setOnAction(event -> onSelectImage(stage));

        // "Classificar" começa desabilitado: só faz sentido depois de escolher
        // uma imagem.
        classifyButton.setDisable(true);
        classifyButton.setOnAction(event -> onClassify());

        HBox buttonsBox = new HBox(10, selectButton, classifyButton);
        buttonsBox.setAlignment(Pos.CENTER);

        // Área de pré-visualização da imagem, com tamanho fixo de 200x200 e
        // proporção preservada (um dígito de 28x28 é ampliado na tela).
        imageView.setFitWidth(200);
        imageView.setFitHeight(200);
        imageView.setPreserveRatio(true);
        imageView.setStyle("-fx-background-color: #202020;");

        // Indicador de "carregando": só aparece enquanto a inferência roda.
        progressIndicator.setVisible(false);
        progressIndicator.setMaxSize(28, 28);

        VBox imageBox = new VBox(10, imageView, progressIndicator);
        imageBox.setAlignment(Pos.CENTER);

        resultsBox.setPadding(new Insets(10, 0, 0, 0));
        resultsBox.setAlignment(Pos.CENTER_LEFT);

        // Centro da janela: imagem em cima, resultados embaixo.
        VBox centerBox = new VBox(20, imageBox, resultsBox);
        centerBox.setAlignment(Pos.TOP_CENTER);
        centerBox.setPadding(new Insets(10));

        // BorderPane divide a janela em topo / centro / rodapé.
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(16));
        root.setTop(buttonsBox);
        root.setCenter(centerBox);
        root.setBottom(statusLabel);
        BorderPane.setAlignment(buttonsBox, Pos.CENTER);
        BorderPane.setAlignment(statusLabel, Pos.CENTER);
        BorderPane.setMargin(buttonsBox, new Insets(0, 0, 16, 0));
        BorderPane.setMargin(statusLabel, new Insets(12, 0, 0, 0));

        // Scene = conteúdo da janela; Stage = a janela em si.
        Scene scene = new Scene(root, 420, 580);
        stage.setTitle("Reconhecimento de Dígitos Manuscritos — DJL + JavaFX");
        stage.setScene(scene);
        // Ao fechar a janela, libera a memória nativa do modelo.
        stage.setOnCloseRequest(event -> {
            if (classifier != null) {
                classifier.close();
            }
        });
        stage.show();
    }

    /** Abre o seletor de arquivos nativo do sistema operacional. */
    private void onSelectImage(Stage stage) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Selecionar imagem do dígito manuscrito");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Imagens (PNG, JPG, BMP)", "*.png", "*.jpg", "*.jpeg", "*.bmp"),
                new FileChooser.ExtensionFilter("Todos os arquivos", "*.*"));

        File file = fileChooser.showOpenDialog(stage);
        if (file == null) {
            return; // usuário cancelou a seleção
        }

        // Guarda o caminho (usado depois por onClassify) e mostra a imagem.
        selectedImagePath = file.toPath();
        imageView.setImage(new Image(file.toURI().toString()));
        // Habilita a classificação e limpa o resultado da imagem anterior.
        classifyButton.setDisable(false);
        resultsBox.getChildren().clear();
        statusLabel.setText("Imagem selecionada: " + file.getName());
    }

    /** Executa a inferência em segundo plano e depois atualiza a interface. */
    private void onClassify() {
        if (selectedImagePath == null) {
            return;
        }

        // Estado "ocupado": evita cliques repetidos e mostra o indicador.
        classifyButton.setDisable(true);
        progressIndicator.setVisible(true);
        statusLabel.setText("Classificando...");
        resultsBox.getChildren().clear();

        // Task roda em outra thread. Se chamássemos classifier.classify()
        // direto aqui, na Application Thread do JavaFX, a janela travaria
        // (ficaria sem responder) durante a inferência.
        Task<Classifications> task = new Task<>() {
            @Override
            protected Classifications call() throws Exception {
                return classifier.classify(selectedImagePath);
            }
        };

        // succeeded/failed já são disparados de volta na Application Thread,
        // então é seguro mexer nos componentes da interface aqui dentro.
        task.setOnSucceeded(event -> {
            showResults(task.getValue());
            progressIndicator.setVisible(false);
            classifyButton.setDisable(false);
            statusLabel.setText("Concluído.");
        });

        task.setOnFailed(event -> {
            progressIndicator.setVisible(false);
            classifyButton.setDisable(false);
            statusLabel.setText("Erro ao classificar a imagem.");
            Throwable ex = task.getException();
            showError("Erro ao classificar", ex == null ? "Erro desconhecido" : String.valueOf(ex.getMessage()));
        });

        // Thread "daemon": não impede o programa de encerrar se a janela for
        // fechada enquanto a inferência ainda está rodando.
        Thread thread = new Thread(task, "digit-classification");
        thread.setDaemon(true);
        thread.start();
    }

    /** Mostra o dígito reconhecido em destaque e a probabilidade de cada classe (0-9). */
    private void showResults(Classifications classifications) {
        resultsBox.getChildren().clear();

        // Dígito de maior probabilidade, em destaque.
        Classifications.Classification best = classifications.best();
        Label bestLabel = new Label(
                "Dígito reconhecido: " + best.getClassName()
                        + "   (" + String.format("%.1f%%", best.getProbability() * 100) + ")");
        bestLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        resultsBox.getChildren().add(bestLabel);

        // Uma linha por dígito (0-9), da mais provável para a menos provável.
        classifications.items().stream()
                .sorted((a, b) -> Double.compare(b.getProbability(), a.getProbability()))
                .forEach(item -> resultsBox.getChildren().add(buildProbabilityRow(item)));
    }

    /** Uma linha "dígito | barra de progresso | percentual" para cada classe. */
    private HBox buildProbabilityRow(Classifications.Classification item) {
        Label digitLabel = new Label(item.getClassName());
        digitLabel.setMinWidth(20);

        ProgressBar bar = new ProgressBar(item.getProbability());
        bar.setMinWidth(220);

        Label percentLabel = new Label(String.format("%.1f%%", item.getProbability() * 100));
        percentLabel.setMinWidth(50);

        HBox row = new HBox(8, digitLabel, bar, percentLabel);
        row.setAlignment(Pos.CENTER_LEFT);
        return row;
    }

    /** Mostra uma caixa de diálogo de erro e espera o usuário fechá-la. */
    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR, message);
        alert.setHeaderText(title);
        alert.showAndWait();
    }

    /**
     * Mostra um erro fatal e encerra o JavaFX. Platform.runLater adia a
     * exibição do diálogo para depois que start() terminar (não é permitido
     * chamar showAndWait() durante o próprio start()).
     */
    private void showFatalErrorAndExit(String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.ERROR, message);
            alert.setHeaderText("Erro ao iniciar a aplicação");
            alert.showAndWait();
            Platform.exit();
        });
    }
}
