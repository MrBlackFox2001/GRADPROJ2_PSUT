import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GRU, Dense, Flatten

def load_and_preprocess_data(s3_uri, local_path='local/path', bucket='psutgrad', key_prefix=''):
    """
    Load MIT-BIH Arrhythmia Database CSV data from an S3 URI, preprocess, and split into training and testing sets.

    Parameters:
    - s3_uri (str): The S3 URI of the MIT-BIH Arrhythmia Database CSV file.
    - local_path (str, optional): The local path to download the data. Defaults to 'local/path'.
    - bucket (str, optional): The S3 bucket name. Defaults to 'psutgrad'.
    - key_prefix (str, optional): The S3 key prefix. Defaults to an empty string.

    Returns:
    - Tuple: Four DataFrames - x_train, x_val, x_test, y_train, y_val, y_test.
    """
    # Load data from S3
    sagemaker_session = sagemaker.Session()
    sagemaker_session.download_data(path=local_path, bucket=bucket, key_prefix=key_prefix)
    ecg_df = pd.read_csv(f'{local_path}/MIT-BIH Arrhythmia Database.csv')

    # Preprocess data
    X = ecg_df.drop(['type', 'record'], axis=1)
    y = lb.fit_transform(ecg_df['type'])
    
    # Handle missing values
    missing_values_report(X)

    # Remove outliers
    ecg_df = remove_outliers(ecg_df)

    # Handle class imbalance
    ecg_df = handle_class_imbalance(ecg_df)

    # Split data into train, validation, and test sets
    x_train, x_test, y_train, y_test = split_data(ecg_df)

    # Apply SMOTE for class imbalance
    x_train_resampled, y_train_resampled = apply_smote(x_train, y_train)

    # Scale data using RobustScaler
    x_train_scaled, x_val_scaled, x_test_scaled = scale_data(x_train_resampled, x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train_resampled, y_test


def create_cnn_gru_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) with Gated Recurrent Unit (GRU) model.

    Parameters:
    - input_shape (tuple): Shape of the input data (excluding batch size).

    Returns:
    - Sequential: Keras Sequential model.
    """
    model = Sequential()
    
    # Add Convolutional layers
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(Dropout(0.2))
    
    # Add GRU layer
    gru_units = 512
    model.add(GRU(gru_units))
    
    # Add Dense layers
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    # Output layer
    model.add(Dense(units=4, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def missing_values_report(df):
    """
    Check and report missing values in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    """
    counter = 0
    for index, row in df.iterrows():
        for col in df.columns:
            if pd.isnull(row[col]):
                counter += 1
                print(f"Missing value in row {index}, column {col}")
    if counter == 0:
        print("No missing values found")


def remove_outliers(df):
    """
    Remove outliers from a DataFrame using the IQR method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[~outliers]

    return df


def handle_class_imbalance(df):
    """
    Handle class imbalance by oversampling the minority class.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with balanced classes.
    """
    mask = df['type'] == 'Q'
    df = df[~mask]

    n_rows = df[df['type'] == 'N']
    rows_to_keep = int(len(n_rows) * 0.20)
    n_rows = n_rows.sample(n=rows_to_keep, random_state=42)

    df = pd.concat([df, n_rows])
    df = df.drop(columns=['record'])

    return df


def split_data(df):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - Tuple: Four DataFrames - x_train, x_test, y_train, y_test.
    """
    X = df.drop('type', axis=1)
    y = df['type']
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def apply_smote(x_train, y_train):
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance class distribution.

    Parameters:
    - x_train (pd.DataFrame): Input features of the training set.
    - y_train (pd.Series): Target labels of the training set.

    Returns:
    - Tuple: Two arrays - x_train_resampled, y_train_resampled.
    """
    smote = SMOTE()
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print("Before SMOTE:")
    print(pd.Series(y_train).value_counts())
    print("After SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    return x_train_resampled, y_train_resampled


def scale_data(x_train, x_test):
    """
    Scale the input data using RobustScaler.

    Parameters:
    - x_train (pd.DataFrame): Input features of the training set.
    - x_test (pd.DataFrame): Input features of the testing set.

    Returns:
    - Tuple: Three arrays - x_train_scaled, x_val_scaled, x_test_scaled.
    """
    robust_scaler = RobustScaler()

    # Fit the scaler on the training data and transform the training data
    x_train_scaled = robust_scaler.fit_transform(x_train)

    # Transform the validation and test data using the same scaler
    x_val_scaled = robust_scaler.transform(x_val)
    x_test_scaled = robust_scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled


def main():
    """
    Main function to demonstrate the use of the provided functions.
    """
    s3_uri_example = 's3://psutgrad/MIT-BIH Arrhythmia Database.csv'
    x_train_scaled, x_val_scaled, x_test_scaled, y_train_resampled, y_test = load_and_preprocess_data(s3_uri_example)

    input_shape = (x_train_scaled.shape[1], 1)
    model = create_cnn_gru_model(input_shape)

    # Train the model
    model.fit(x_train_scaled, y_train_resampled, epochs=10, validation_data=(x_val_scaled, y_val))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
