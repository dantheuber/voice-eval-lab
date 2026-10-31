import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import * as Tooltip from "@radix-ui/react-tooltip";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { ensureTransport, ApiError } from "./api/transport";
import "./styles/base.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (count, err) => !(err instanceof ApiError && (err.status === 404 || err.status === 400)) && count < 2,
      staleTime: 5_000,
      refetchOnWindowFocus: false,
    },
  },
});

ensureTransport().then(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <Tooltip.Provider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </Tooltip.Provider>
      </QueryClientProvider>
    </React.StrictMode>,
  );
});
