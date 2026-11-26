import React from 'react';
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="bg-blue-500 text-white p-4">
      <div className="max-w-4xl mx-auto flex justify-between">
        <Link to="/journal" className="hover:underline">
          Journal
        </Link>
        <Link to="/dashboard" className="hover:underline">
          Dashboard
        </Link>
      </div>
    </nav>
  );
}
