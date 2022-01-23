
from argparse import ArgumentParser
from Auto_Encoder.AEncoder import AEED, load_AEED
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_file(path_test_file):
    autoencoder = load_AEED('Models/autoencoder.json','Models/autoencoder.h5')
    df_test = pd.read_csv(path_test_file)
    scaler = MinMaxScaler()
    sensor_cols = [col for col in df_test.columns if col not in ['INDEX(TIME_IN_HOURS)']]
    X_test = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.fit_transform(df_test[sensor_cols]))
    _, validation_errors = autoencoder.predict(X_test)
    theta = validation_errors.mean(axis = 1).quantile(0.995)
    y_pred, _ =  autoencoder.detect(X_test, theta = theta , window = 3, average=True)
    df_test["LABEL"] = y_pred
    result  = pd.DataFrame()
    result["INDEX(TIME_IN_HOURS)"] = df_test["INDEX(TIME_IN_HOURS)"]
    result["LABEL"] = df_test["LABEL"].apply(lambda x: "ATTACK" if x ==True else "NORMAL" )

    result.to_csv("result.csv",index=False)

    
    print("Processed %s file and saved to ./result.csv" % path_test_file)
    return

def main():
    parser = ArgumentParser(prog='IDS_test')
    parser.add_argument('path_test_file', help="Test file path")
    args = parser.parse_args()
    print("Processing %s file for IDS" % args.path_test_file)
    process_file(args.path_test_file)

if __name__ == '__main__':
    main()