import numpy as np

from predict_mouse.components.logisticgridpredictor import LogisticGridPredictor


def parse_motion_data_line(line):
    flat_curve = np.fromstring(line.strip(), sep=',')
    return flat_curve.reshape((int(flat_curve.size / 2), 2))


def main():

    with open('mouse_train_data.csv') as f:
        content = f.readlines()

    motion_data = [parse_motion_data_line(curve) for curve in content]

    predictor = LogisticGridPredictor(7, 7, 800, 600, 0.5, 25)
    predictor.set_train_data(motion_data)


if __name__ == '__main__':
    main()
