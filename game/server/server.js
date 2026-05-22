const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

const players = {};

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  players[socket.id] = {
    id: socket.id,
    position: { x: 0, y: 1.7, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    room: 'public'
  };

  socket.on('joinRoom', (roomId) => {
    // Leave current room
    socket.leave(players[socket.id].room);
    
    // Join new room
    players[socket.id].room = roomId;
    socket.join(roomId);
    
    console.log(`User ${socket.id} joined room: ${roomId}`);
    
    // Send current players in THAT room only
    const playersInRoom = {};
    Object.values(players).forEach(p => {
      if (p.room === roomId) playersInRoom[p.id] = p;
    });
    
    socket.emit('currentPlayers', playersInRoom);
    socket.to(roomId).emit('newPlayer', players[socket.id]);
  });

  socket.on('playerMovement', (movementData) => {
    if (players[socket.id]) {
      players[socket.id].position = movementData.position;
      players[socket.id].rotation = movementData.rotation;
      socket.to(players[socket.id].room).emit('playerMoved', players[socket.id]);
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
    const roomId = players[socket.id]?.room;
    delete players[socket.id];
    io.to(roomId).emit('playerDisconnected', socket.id);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
