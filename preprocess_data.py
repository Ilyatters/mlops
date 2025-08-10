from minio import Minio
from minio.error import S3Error
import dask.dataframe as dd
import io

bucket_name = 'ml-data'
object_name = 'train_datagen'

bucket_name_features = 'ml-data'
object_name_features= 'features'

minio_client = Minio(
    'localhost:9000',
    access_key='admin',
    secret_key='adminadmin',
    secure=False
)

try:
    minio_client.fget_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path='/Users/ilyatarasevich/Desktop/Project_MLOps/data/train_datagen.csv'
    )
    print('Файл успешно скачан')

except S3Error as e:
    print(f'Ошибка {e}')

raw_data = '/Users/ilyatarasevich/Desktop/Project_MLOps/data/train_datagen.csv' \


ddf = dd.read_csv(raw_data)
ddf = ddf.dropna()
ddf = ddf.drop_duplicates()
ddf.info()


try:
    if not minio_client.bucket_exists(bucket_name_features):
        minio_client.make_bucket(bucket_name_features)
except Exception as e:
    print(f"Ошибка при работе с бакетом: {str(e)}")
    raise

csv_data = ddf.compute().to_csv(index=False, header=True).encode('utf-8')

try:
    minio_client.put_object(
        bucket_name_features,
        object_name_features,
        io.BytesIO(csv_data),
        length=len(csv_data),
        content_type='text/csv'
    )
    print(f"Файл {object_name_features} загружен в MinIO с Content-Type: text/csv")
except Exception as e:
    print(f"Ошибка при загрузке файла: {str(e)}")
    raise
