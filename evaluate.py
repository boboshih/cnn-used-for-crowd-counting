from model import load_train_set
import matplotlib.pyplot as plt
import keras


fig_count = 1


def evaluate(folder_path, model_path):
    model = keras.models.load_model('0107200137_5e-07_both_8_model')

    X, y = load_train_set(folder_path)

    print(y.std())
    result = model.evaluate(X, y, batch_size=8)
    result = dict(zip(model.metrics_names, result))
    print(result)

    pre = model.predict(X)

    plt.figure(figsize=(18, 8))
    plt.plot(y, 'b.', label='actual')
    plt.plot(pre, 'gx', label='pred')
    plt.title('mse: ' + str(result['mse']) + '    mae: ' + str(result['mae']))
    plt.legend()
    global fig_count
    plt.savefig('./log/' + model_path.split('/')[-1].replace('_model', '_') + 'evaluate_' + str(fig_count) + '.png')
    fig_count += 1


if __name__ == '__main__':
    model_path = './model/0107200137_5e-07_both_8_model'
    evaluate(['./part_A_final/test_data/'], model_path)
    evaluate(['./part_B_final/test_data/'], model_path)
