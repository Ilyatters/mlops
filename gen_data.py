import dask.dataframe as dd
import numpy as np
import pandas as pd
from minio import Minio
import io

ddf = dd.from_pandas(
    pd.DataFrame({
        'x': np.random.rand(10000) * 100
    }), npartitions=1
)

ddf['y'] = ddf['x'] * 3

# Инициализация клиента MinIO
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="adminadmin",
    secure=False
)

bucket_name = 'ml-data'
object_name = 'train_datagen'

try:
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
except Exception as e:
    print(f"Ошибка при работе с бакетом: {str(e)}")
    raise

csv_data = ddf.compute().to_csv(index=False, header=True).encode('utf-8')

try:
    minio_client.put_object(
        bucket_name,
        object_name,
        io.BytesIO(csv_data),
        length=len(csv_data),
        content_type='text/csv'
    )
    print(f"Файл {object_name} загружен в MinIO с Content-Type: text/csv")
except Exception as e:
    print(f"Ошибка при загрузке файла: {str(e)}")
    raise