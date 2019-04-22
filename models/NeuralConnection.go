package models

type NeuralConnection struct {
	UUID string `json:"uuid" form:"uuid" query:"uuid"`

	ToNodeUUID string `json:"to_node_uuid" form:"to_node_uuid" query:"to_node_uuid"`

	FromNodeUUID string `json:"from_node_uuid" form:"from_node_uuid" query:"from_node_uuid"`

	Weight float64 `json:"weight" form:"weight" query:"weight"`
}
