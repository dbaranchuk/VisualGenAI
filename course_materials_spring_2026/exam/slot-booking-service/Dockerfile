FROM node:20-alpine
WORKDIR /app
COPY . .
ENV PORT=3000
ENV BOOKINGS_FILE=/data/bookings.json
EXPOSE 3000
CMD ["node", "server.js"]
