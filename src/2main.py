import Setup
from Network_Store_Module import (
    create_project, add_labeled_data, create_train_dataset, train_network,
)

project_config = create_project(Setup.NETWORK_STORE, name="mitten_tracker")

add_labeled_data(project_config, Setup.STORE + r"\labeled-data\HDMI-A__default")
# add_labeled_data(project_config, Setup.STORE + r"\labeled-data\HDMI-A__dense")

create_train_dataset(project_config)
train_network(project_config, epochs=600)