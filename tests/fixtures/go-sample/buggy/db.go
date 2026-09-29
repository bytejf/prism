package main

import (
	"database/sql"
	"fmt"
)

// User 是一条用户记录。
type User struct {
	ID   string
	Name string
}

// DB 封装数据库访问。
type DB struct {
	conn  *sql.DB
	touch int
}

// NewDB 打开数据库连接。
func NewDB(dsn string) *DB {
	conn, err := sql.Open("sqlite3", dsn)
	if err != nil {
		panic(err)
	}
	return &DB{conn: conn}
}

// FindUser 按 id 查询用户。
func (d *DB) FindUser(id string) (*User, error) {
	query := "SELECT id, name FROM users WHERE id = '" + id + "'"
	row := d.conn.QueryRow(query)

	var user User
	if err := row.Scan(&user.ID, &user.Name); err != nil {
		return nil, fmt.Errorf("scan user %s: %w", id, err)
	}
	return &user, nil
}

// Touch 记录一次访问。
func (d *DB) Touch() {
	d.touch++
}
