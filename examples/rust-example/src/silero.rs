use crate::utils;
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use std::path::Path;

#[derive(Debug)]
pub struct Silero {
    session: ort::Session,
    sample_rate: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    state: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    context: Array1<f32>,
    context_size: usize,
}

impl Silero {
    pub fn new(
        sample_rate: utils::SampleRate,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, ort::Error> {
        let session = ort::Session::builder()?.commit_from_file(model_path)?;
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let sample_rate_val: i64 = sample_rate.into();
        let context_size = if sample_rate_val == 16000 { 64 } else { 32 };
        let context = Array1::<f32>::zeros(context_size);
        let sample_rate = Array::from_shape_vec([1], vec![sample_rate_val]).unwrap();
        Ok(Self {
            session,
            sample_rate,
            state,
            context,
            context_size,
        })
    }

    pub fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.context = Array1::<f32>::zeros(self.context_size);
    }

    pub fn calc_level(&mut self, audio_frame: &[i16]) -> Result<f32, ort::Error> {
        let data = audio_frame
            .iter()
            .map(|x| (*x as f32) / (i16::MAX as f32))
            .collect::<Vec<_>>();

        // Concatenate context with input
        let mut input_with_context = Vec::with_capacity(self.context_size + data.len());
        input_with_context.extend_from_slice(self.context.as_slice().unwrap());
        input_with_context.extend_from_slice(&data);

        let frame = Array2::<f32>::from_shape_vec([1, input_with_context.len()], input_with_context).unwrap();

        let inps = ort::inputs![
            frame,
            std::mem::take(&mut self.state),
            self.sample_rate.clone(),
        ]?;
        let res = self
            .session
            .run(ort::SessionInputs::ValueSlice::<3>(&inps))?;

        self.state = res["stateN"].try_extract_tensor().unwrap().to_owned();

        // Update context with last context_size samples from the input
        if data.len() >= self.context_size {
            self.context = Array1::from_vec(data[data.len() - self.context_size..].to_vec());
        }

        let prob = *res["output"]
            .try_extract_raw_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap();
        Ok(prob)
    }
}
