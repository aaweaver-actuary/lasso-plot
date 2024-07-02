# Serve the next.js app
FROM node:16.13.0-alpine


# Set the working directory
WORKDIR /app

# Copy the prettier config file
COPY ../.prettierrc .prettierrc

# Install the dependencies
RUN npm install

# Build the app if NEXTJS_DEV_MODE=1
RUN if [ "$NEXTJS_DEV_MODE" = "1" ]; then npm run build; fi

# Expose the port
EXPOSE 3000

# Run the app in either dev or production mode depending on the NEXTJS_DEV_MODE environment variable
CMD if [ "$NEXTJS_DEV_MODE" = "1" ]; then npm run dev; else npm run start; fi