import flwr as fl

from flower_app.task import (
    create_model,
    get_client_data,
    train_client
)

from models.parameters import (
    get_parameters,
    set_parameters
)

from flwr.app import (
    ArrayRecord,
    MetricRecord,
    RecordDict,
    Message
)


app = fl.client.ClientApp()


@app.train()
def train(msg: Message, context):

    client_id = int(
        context.node_config["partition-id"]
    )

    model = create_model()

    X, y = get_client_data(client_id)

    # Receive global model
    parameters = msg.content["arrays"]

    parameter_arrays = (
        parameters.to_numpy_ndarrays()
    )

    set_parameters(
        model,
        parameter_arrays
    )

    # Save global parameters BEFORE local training
    global_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    # Read configuration sent by server
    # config_record = msg.content["config"]

    # algorithm = config_record.get(
    #     "algorithm",
    #     "fedavg"
    # )

    # proximal_mu = float(
    #     config_record.get(
    #         "proximal-mu",
    #         0.0
    #     )
    # )


    config_record = msg.content["config"]

    if "proximal-mu" in config_record:

        algorithm = "fedprox"

        proximal_mu = float(
            config_record["proximal-mu"]
        )

    else:

        algorithm = "fedavg"

        proximal_mu = 0.0

    # Local training
    train_client(
        model,
        X,
        y,
        algorithm=algorithm,
        proximal_mu=proximal_mu,
        global_parameters=global_parameters
    )

    # Send updated model back
    updated_parameters = get_parameters(model)

    model_record = ArrayRecord(
        updated_parameters
    )

    metric_record = MetricRecord({
        "num-examples": len(X)
    })

    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record
    })

    return Message(
        content=content,
        reply_to=msg
    )