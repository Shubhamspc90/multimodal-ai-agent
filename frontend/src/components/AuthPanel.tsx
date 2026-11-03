import { useState, type FormEvent } from "react";
import {
  loginUser,
  registerUser,
} from "../lib/api";
import type { AuthUser } from "../lib/api";
import "./AuthPanel.css";

type AuthMode = "login" | "register";

interface AuthPanelProps {
  onAuthenticated: (user: AuthUser) => void;
  initialError?: string;
}

export default function AuthPanel({
  onAuthenticated,
  initialError = "",
}: AuthPanelProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(initialError);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (isSubmitting) return;

    setError("");
    setIsSubmitting(true);

    try {
      if (mode === "register") {
        await registerUser({
          name: name.trim(),
          email: email.trim(),
          password,
        });
      }

      // Registration creates the account. Login establishes the session
      // and sets the authentication and CSRF cookies.
      const result = await loginUser({
        email: email.trim(),
        password,
      });

      onAuthenticated(result.user);
    } catch (submitError: unknown) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Unable to authenticate. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const changeMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    setError("");
    setPassword("");
  };

  return (
    <main className="auth-screen">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-brand-mark">✦</div>

        <p className="auth-eyebrow">MULTIMODAL AI AGENT</p>

        <h1 id="auth-title">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h1>

        <p className="auth-description">
          {mode === "login"
            ? "Sign in to continue to your AI workspace."
            : "Create an account to start using your personal AI workspace."}
        </p>

        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === "register" && (
            <label>
              <span>Full name</span>
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Enter your name"
                autoComplete="name"
                maxLength={100}
                required
                disabled={isSubmitting}
              />
            </label>
          )}

          <label>
            <span>Email address</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              required
              disabled={isSubmitting}
            />
          </label>

          <label>
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              autoComplete={
                mode === "login" ? "current-password" : "new-password"
              }
              minLength={8}
              maxLength={128}
              required
              disabled={isSubmitting}
            />
          </label>

          <button
            className="auth-submit"
            type="submit"
            disabled={isSubmitting}
          >
            {isSubmitting
              ? "Please wait..."
              : mode === "login"
                ? "Sign in"
                : "Create account"}
          </button>
        </form>

        <div className="auth-switch">
          {mode === "login" ? (
            <>
              <span>Don't have an account?</span>
              <button
                type="button"
                onClick={() => changeMode("register")}
                disabled={isSubmitting}
              >
                Sign up
              </button>
            </>
          ) : (
            <>
              <span>Already have an account?</span>
              <button
                type="button"
                onClick={() => changeMode("login")}
                disabled={isSubmitting}
              >
                Sign in
              </button>
            </>
          )}
        </div>
      </section>
    </main>
  );
}