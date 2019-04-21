package models

type NeuralConnection struct {
	UUID string `json:"uuid" form:"uuid" query:"uuid"`

	ToNodeUUID string `json:"uuid" form:"uuid" query:"uuid"`

	FromNodeUUID string `json:"uuid" form:"uuid" query:"uuid"`

	Weight float64 `json:"weight" form:"weight" query:"weight"`
}
