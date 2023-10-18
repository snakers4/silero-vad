package org.example;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SlieroVadOnnxModel {
    // Define private variable OrtSession
    private final OrtSession session;
    private float[][][] h;
    private float[][][] c;
    // Define the last sample rate
    private int lastSr = 0;
    // Define the last batch size
    private int lastBatchSize = 0;
    // Define a list of supported sample rates
    private static final List<Integer> SAMPLE_RATES = Arrays.asList(8000, 16000);

    // Constructor
    public SlieroVadOnnxModel(String modelPath) throws OrtException {
        // Get the ONNX runtime environment
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        // Create an ONNX session options object
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // Set the InterOp thread count to 1, InterOp threads are used for parallel processing of different computation graph operations
        opts.setInterOpNumThreads(1);
        // Set the IntraOp thread count to 1, IntraOp threads are used for parallel processing within a single operation
        opts.setIntraOpNumThreads(1);
        // Add a CPU device, setting to false disables CPU execution optimization
        opts.addCPU(true);
        // Create an ONNX session using the environment, model path, and options
        session = env.createSession(modelPath, opts);
        // Reset states
        resetStates();
    }

    /**
     * Reset states
     */
    void resetStates() {
        h = new float[2][1][64];
        c = new float[2][1][64];
        lastSr = 0;
        lastBatchSize = 0;
    }

    public void close() throws OrtException {
        session.close();
    }

    /**
     * Define inner class ValidationResult
     */
    public static class ValidationResult {
        public final float[][] x;
        public final int sr;

        // Constructor
        public ValidationResult(float[][] x, int sr) {
            this.x = x;
            this.sr = sr;
        }
    }

    /**
     * Function to validate input data
     */
    private ValidationResult validateInput(float[][] x, int sr) {
        // Process the input data with dimension 1
        if (x.length == 1) {
            x = new float[][]{x[0]};
        }
        // Throw an exception when the input data dimension is greater than 2
        if (x.length > 2) {
            throw new IllegalArgumentException("Incorrect audio data dimension: " + x[0].length);
        }

        // Process the input data when the sample rate is not equal to 16000 and is a multiple of 16000
        if (sr != 16000 && (sr % 16000 == 0)) {
            int step = sr / 16000;
            float[][] reducedX = new float[x.length][];

            for (int i = 0; i < x.length; i++) {
                float[] current = x[i];
                float[] newArr = new float[(current.length + step - 1) / step];

                for (int j = 0, index = 0; j < current.length; j += step, index++) {
                    newArr[index] = current[j];
                }

                reducedX[i] = newArr;
            }

            x = reducedX;
            sr = 16000;
        }

        // If the sample rate is not in the list of supported sample rates, throw an exception
        if (!SAMPLE_RATES.contains(sr)) {
            throw new IllegalArgumentException("Only supports sample rates " + SAMPLE_RATES + " (or multiples of 16000)");
        }

        // If the input audio block is too short, throw an exception
        if (((float) sr) / x[0].length > 31.25) {
            throw new IllegalArgumentException("Input audio is too short");
        }

        // Return the validated result
        return new ValidationResult(x, sr);
    }

    /**
     * Method to call the ONNX model
     */
    public float[] call(float[][] x, int sr) throws OrtException {
        ValidationResult result = validateInput(x, sr);
        x = result.x;
        sr = result.sr;

        int batchSize = x.length;

        if (lastBatchSize == 0 || lastSr != sr || lastBatchSize != batchSize) {
            resetStates();
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OnnxTensor inputTensor = null;
        OnnxTensor hTensor = null;
        OnnxTensor cTensor = null;
        OnnxTensor srTensor = null;
        OrtSession.Result ortOutputs = null;

        try {
            // Create input tensors
            inputTensor = OnnxTensor.createTensor(env, x);
            hTensor = OnnxTensor.createTensor(env, h);
            cTensor = OnnxTensor.createTensor(env, c);
            srTensor = OnnxTensor.createTensor(env, new long[]{sr});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input", inputTensor);
            inputs.put("sr", srTensor);
            inputs.put("h", hTensor);
            inputs.put("c", cTensor);

            // Call the ONNX model for calculation
            ortOutputs = session.run(inputs);
            // Get the output results
            float[][] output = (float[][]) ortOutputs.get(0).getValue();
            h = (float[][][]) ortOutputs.get(1).getValue();
            c = (float[][][]) ortOutputs.get(2).getValue();

            lastSr = sr;
            lastBatchSize = batchSize;
            return output[0];
        } finally {
            if (inputTensor != null) {
                inputTensor.close();
            }
            if (hTensor != null) {
                hTensor.close();
            }
            if (cTensor != null) {
                cTensor.close();
            }
            if (srTensor != null) {
                srTensor.close();
            }
            if (ortOutputs != null) {
                ortOutputs.close();
            }
        }
    }
}
