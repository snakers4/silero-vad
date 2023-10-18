package org.example;

import ai.onnxruntime.OrtException;
import javax.sound.sampled.*;
import java.util.Map;

public class App {

    private static final String MODEL_PATH = "src/main/resources/silero_vad.onnx";
    private static final int SAMPLE_RATE = 16000;
    private static final float START_THRESHOLD = 0.6f;
    private static final float END_THRESHOLD = 0.45f;
    private static final int MIN_SILENCE_DURATION_MS = 600;
    private static final int SPEECH_PAD_MS = 500;
    private static final int WINDOW_SIZE_SAMPLES = 2048;

    public static void main(String[] args) {
        // Initialize the Voice Activity Detector
        SlieroVadDetector vadDetector;
        try {
            vadDetector = new SlieroVadDetector(MODEL_PATH, START_THRESHOLD, END_THRESHOLD, SAMPLE_RATE, MIN_SILENCE_DURATION_MS, SPEECH_PAD_MS);
        } catch (OrtException e) {
            System.err.println("Error initializing the VAD detector: " + e.getMessage());
            return;
        }

        // Set audio format
        AudioFormat format = new AudioFormat(SAMPLE_RATE, 16, 1, true, false);
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

        // Get the target data line and open it with the specified format
        TargetDataLine targetDataLine;
        try {
            targetDataLine = (TargetDataLine) AudioSystem.getLine(info);
            targetDataLine.open(format);
            targetDataLine.start();
        } catch (LineUnavailableException e) {
            System.err.println("Error opening target data line: " + e.getMessage());
            return;
        }

        // Main loop to continuously read data and apply Voice Activity Detection
        while (targetDataLine.isOpen()) {
            byte[] data = new byte[WINDOW_SIZE_SAMPLES];

            int numBytesRead = targetDataLine.read(data, 0, data.length);
            if (numBytesRead <= 0) {
                System.err.println("Error reading data from target data line.");
                continue;
            }

            // Apply the Voice Activity Detector to the data and get the result
            Map<String, Double> detectResult;
            try {
                detectResult = vadDetector.apply(data, true);
            } catch (Exception e) {
                System.err.println("Error applying VAD detector: " + e.getMessage());
                continue;
            }

            if (!detectResult.isEmpty()) {
                System.out.println(detectResult);
            }
        }

        // Close the target data line to release audio resources
        targetDataLine.close();
    }
}
