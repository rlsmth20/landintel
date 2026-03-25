FROM node:20

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY package*.json ./

RUN npm ci

COPY . .

RUN rm -rf .next && npm run build

CMD ["npm", "start"]
