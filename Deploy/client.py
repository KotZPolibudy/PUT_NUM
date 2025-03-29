import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(
        input_data=[
            [5.9, 3, 5.1, 1.8]
        ],
    )
    print(result)