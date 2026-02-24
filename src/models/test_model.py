from nilm_cnn import build_nilm_cnn

if __name__ == "__main__":
    model = build_nilm_cnn(window_size=50)
    model.summary()
