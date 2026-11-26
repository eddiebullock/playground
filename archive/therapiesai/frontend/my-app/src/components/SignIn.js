import React, { useState } from 'react';
import { auth } from '../firebase';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSignIn = async (e) => {
    e.preventDefault();
    try {
      await signInWithEmailAndPassword(auth, email, password);
      console.log('User signed in successfully!');
      navigate('/dashboard'); // Redirect to the dashboard
    } catch (err) {
      console.error(err.message);
      setError('Failed to sign in. Please check your credentials.');
    }
  };

  return (
    <div className="flex flex-col justify-center items-center h-screen bg-gray-50 p-4">
      <form
        onSubmit={handleSignIn}
        className="bg-white p-6 rounded-lg shadow-md w-full max-w-sm"
      >
        <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">
          Sign In
        </h2>
        {error && <p className="text-red-500 text-sm">{error}</p>}
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-3 mb-3 border rounded-lg text-gray-800"
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-3 mb-3 border rounded-lg text-gray-800"
          required
        />
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600"
        >
          Sign In
        </button>
        <p className="mt-3 text-sm text-center">
          Don’t have an account?{' '}
          <a href="/signup" className="text-blue-500 hover:underline">
            Create one here
          </a>
        </p>
      </form>
    </div>
  );
}
