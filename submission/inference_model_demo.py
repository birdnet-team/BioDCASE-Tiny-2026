# --
# inference model demo


class InferenceModelDemo(InferenceModelBase):
  """
  inference model base - use this template for your submission model
  """

  def infer_wav_audio(self, wav):
    """
    infer wav audio
    """

    # feature extraction
    features = self.feature_extraction_function(wav)

    # post processing
    features = self.post_process_features(features)

    # model inference
    prediction = self.model.predict()


  def post_process_features(self, features):
    """
    features post processing
    """

    # float32
    features = features.astype(np.float32)

    # transposed
    features = features.T

    return features



if __name__ == '__main__':
  """
  inference model demo
  """

  # inference model demo
  inference_model_demo = InferenceModelDemo(model, feature_extraction_function)