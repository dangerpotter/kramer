# Build stage - using regular node image instead of alpine to avoid symlink issues
FROM node:18 as build

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application using node directly to bypass symlink issues
RUN node node_modules/vite/bin/vite.js build

# Production stage - keep nginx on alpine for smaller image
FROM nginx:alpine

# Install wget for health checks
RUN apk add --no-cache wget

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
