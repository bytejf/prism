package main

import (
	"log"
	"net/http"
)

func main() {
	svc := NewService(NewDB("users.db"))
	svc.Run()
	defer svc.Stop()

	handler := NewHandler(svc)

	mux := http.NewServeMux()
	mux.HandleFunc("/user", handler.GetUser)

	log.Println("listening on :8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatal(err)
	}
}
