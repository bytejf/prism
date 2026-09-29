package main

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Service 封装用户相关的业务逻辑。
type Service struct {
	db   *DB
	mu   sync.Mutex
	stop chan struct{}
}

// NewService 创建 Service。
func NewService(db *DB) *Service {
	return &Service{db: db, stop: make(chan struct{})}
}

// Process 加载用户数据并返回处理结果。
func (s *Service) Process(id string) (string, error) {
	user, err := s.db.FindUser(id)
	if err != nil {
		return "", fmt.Errorf("process %s: %w", id, err)
	}
	return s.handle(user)
}

// Run 启动后台清理任务，收到 stop 信号后退出。
func (s *Service) Run() {
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-s.stop:
				return
			case <-ticker.C:
				s.doWork()
			}
		}
	}()
}

// Stop 通知后台任务退出。
func (s *Service) Stop() {
	close(s.stop)
}

// Lookup 按 id 查询用户。
func (s *Service) Lookup(id string) (*User, error) {
	return s.db.FindUser(id)
}

func (s *Service) handle(user *User) (string, error) {
	if user == nil {
		return "", errors.New("nil user")
	}
	return "processed:" + user.Name, nil
}

func (s *Service) doWork() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.db.Touch()
}
