package com.example.mnist.ui;

/**
 * Ponto de entrada real do JAR executável (é esta classe que vai no
 * Main-Class do manifesto, não a InferenceApp).
 *
 * Isso é proposital: quando se roda "java -jar app.jar" e a classe apontada
 * como Main-Class no manifesto é uma subclasse direta de
 * javafx.application.Application, o launcher do Java recusa a execução com
 * o erro:
 *
 *   Error: JavaFX runtime components are missing, and are required
 *   to run this application
 *
 * mesmo com o JavaFX presente dentro do próprio JAR. Isso acontece porque,
 * ao detectar que a classe principal estende Application, o launcher exige
 * que o JavaFX esteja num module-path nomeado — o que não é o caso de um
 * "jar com dependências" (uber jar) rodando em classpath simples.
 *
 * A solução padrão é usar uma classe intermediária (esta aqui), que NÃO
 * estende Application e apenas delega para InferenceApp.main(...). Assim o
 * launcher não faz a checagem e o JavaFX é carregado normalmente como uma
 * dependência comum do classpath.
 */
public final class Launcher {

    private Launcher() {}

    public static void main(String[] args) {
        // Apenas delega: InferenceApp.main() chama Application.launch(), que
        // inicia o JavaFX e abre a janela.
        InferenceApp.main(args);
    }
}
