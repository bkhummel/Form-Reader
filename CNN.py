from keras.models import model_from_json


class CNN:
    def __init__(self, a):
        self.a = a

    def load_model(self):
        json_file = open('Model/digits_model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights("Model/digits_model.h5")
        print("Loaded model from disk")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

