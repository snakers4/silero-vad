package org.example;

import ai.onnxruntime.OrtException;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class SlieroVadDetector {
    // OnnxModel model used for speech processing
    private final SlieroVadOnnxModel model;
    // Threshold for speech start
    private final float startThreshold;
    // Threshold for speech end
    private final float endThreshold;
    // Sampling rate
    private final int samplingRate;
    // Minimum number of silence samples to determine the end threshold of speech
    private final float minSilenceSamples;
    // Additional number of samples for speech start or end to calculate speech start or end time
    private final float speechPadSamples;
    // Whether in the triggered state (i.e. whether speech is being detected)
    private boolean triggered;
    // Temporarily stored number of speech end samples
    private int tempEnd;
    // Number of samples currently being processed
    private int currentSample;


    public SlieroVadDetector(String modelPath,
                             float startThreshold,
                             float endThreshold,
                             int samplingRate,
                             int minSilenceDurationMs,
                             int speechPadMs) throws OrtException {
        // Check if the sampling rate is 8000 or 16000, if not, throw an exception
        if (samplingRate != 8000 && samplingRate != 16000) {
            throw new IllegalArgumentException("does not support sampling rates other than [8000, 16000]");
        }

        // Initialize the parameters
        this.model = new SlieroVadOnnxModel(modelPath);
        this.startThreshold = startThreshold;
        this.endThreshold = endThreshold;
        this.samplingRate = samplingRate;
        this.minSilenceSamples = samplingRate * minSilenceDurationMs / 1000f;
        this.speechPadSamples = samplingRate * speechPadMs / 1000f;
        // Reset the state
        reset();
    }

    // Method to reset the state, including the model state, trigger state, temporary end time, and current sample count
    public void reset() {
        model.resetStates();
        triggered = false;
        tempEnd = 0;
        currentSample = 0;
    }

    // apply method for processing the audio array, returning possible speech start or end times
    public Map<String, Double> apply(byte[] data, boolean returnSeconds) {

        // Convert the byte array to a float array
        float[] audioData = new float[data.length / 2];
        for (int i = 0; i < audioData.length; i++) {
            audioData[i] = ((data[i * 2] & 0xff) | (data[i * 2 + 1] << 8)) / 32767.0f;
        }

        // Get the length of the audio array as the window size
        int windowSizeSamples = audioData.length;
        // Update the current sample count
        currentSample += windowSizeSamples;

        // Call the model to get the prediction probability of speech
        float speechProb = 0;
        try {
            speechProb = model.call(new float[][]{audioData}, samplingRate)[0];
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        // If the speech probability is greater than the threshold and the temporary end time is not 0, reset the temporary end time
        // This indicates that the speech duration has exceeded expectations and needs to recalculate the end time
        if (speechProb >= startThreshold && tempEnd != 0) {
            tempEnd = 0;
        }

        // If the speech probability is greater than the threshold and not in the triggered state, set to triggered state and calculate the speech start time
        if (speechProb >= startThreshold && !triggered) {
            triggered = true;
            int speechStart = (int) (currentSample - speechPadSamples);
            speechStart = Math.max(speechStart, 0);
            Map<String, Double> result = new HashMap<>();
            // Decide whether to return the result in seconds or sample count based on the returnSeconds parameter
            if (returnSeconds) {
                double speechStartSeconds = speechStart / (double) samplingRate;
                double roundedSpeechStart = BigDecimal.valueOf(speechStartSeconds).setScale(1, RoundingMode.HALF_UP).doubleValue();
                result.put("start", roundedSpeechStart);
            } else {
                result.put("start", (double) speechStart);
            }

            return result;
        }

        // If the speech probability is less than a certain threshold and in the triggered state, calculate the speech end time
        if (speechProb < endThreshold && triggered) {
            // Initialize or update the temporary end time
            if (tempEnd == 0) {
                tempEnd = currentSample;
            }
            // If the number of silence samples between the current sample and the temporary end time is less than the minimum silence samples, return null
            // This indicates that it is not yet possible to determine whether the speech has ended
            if (currentSample - tempEnd < minSilenceSamples) {
                return Collections.emptyMap();
            } else {
                // Calculate the speech end time, reset the trigger state and temporary end time
                int speechEnd = (int) (tempEnd + speechPadSamples);
                tempEnd = 0;
                triggered = false;
                Map<String, Double> result = new HashMap<>();

                if (returnSeconds) {
                    double speechEndSeconds = speechEnd / (double) samplingRate;
                    double roundedSpeechEnd = BigDecimal.valueOf(speechEndSeconds).setScale(1, RoundingMode.HALF_UP).doubleValue();
                    result.put("end", roundedSpeechEnd);
                } else {
                    result.put("end", (double) speechEnd);
                }
                return result;
            }
        }

        // If the above conditions are not met, return null by default
        return Collections.emptyMap();
    }

    public void close() throws OrtException {
        reset();
        model.close();
    }
}
