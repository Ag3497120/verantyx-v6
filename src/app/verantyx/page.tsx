'use client';

import { useEffect } from 'react';

/**
 * Engine page abolished as a primary destination.
 * Redirect to the Verantyx-CLI product page.
 */
export default function VerantyxRedirect() {
  useEffect(() => {
    window.location.replace('/verantyx-cli/');
  }, []);

  return (
    <main
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#050508',
        color: '#94a3b8',
        fontFamily: 'system-ui, sans-serif',
        padding: 24,
        textAlign: 'center',
      }}
    >
      <p>
        Redirecting to{' '}
        <a href="/verantyx-cli/" style={{ color: '#0EA5E9' }}>
          Verantyx-CLI
        </a>
        …
      </p>
    </main>
  );
}
