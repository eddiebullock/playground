import React, { useEffect, useState } from 'react';
import { db, auth } from '../firebase';
import { collection, getDocs, query, where, orderBy, doc, getDoc } from 'firebase/firestore'; // Added missing imports
import { Line } from 'react-chartjs-2';
import { Doughnut } from 'react-chartjs-2';
import 'chart.js/auto';
import 'chartjs-adapter-date-fns';
import TrendVisualizer from './TrendVisualizer'; // Adjust the path as necessary

export default function Dashboard() {
  const [entries, setEntries] = useState([]);
  const [happinessTrend, setHappinessTrend] = useState([]);
  const [averageThisWeek, setAverageThisWeek] = useState(0);
  const [changeFromLastWeek, setChangeFromLastWeek] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streak, setStreak] = useState(0);

  useEffect(() => {
    const fetchStreak = async () => {
      const user = auth.currentUser;
      if (!user) {
        console.error("No user is logged in.");
        return;
      }

      try {
        const userRef = doc(db, "users", user.uid);
        const userSnap = await getDoc(userRef);
        if (userSnap.exists()) {
          setStreak(userSnap.data().streak || 0); // Fetch and set streak
        }
      } catch (error) {
        console.error("Error fetching streak:", error);
      }
    };

    const fetchEntries = async () => {
      const user = auth.currentUser;
      if (!user) {
        console.error("No user is logged in.");
        return;
      }

      const journalRef = collection(db, "journals");
      const q = query(journalRef, where("userId", "==", user.uid), orderBy("timestamp", "desc"));
      const snapshot = await getDocs(q);
      const data = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
        themes: doc.data().themes || [],
        sentiment: doc.data().sentiment || "neutral",
        timestamp: doc.data().timestamp.toDate(),
      }));

      setEntries(data);

      const now = new Date();
      const startOfWeek = new Date(now.setDate(now.getDate() - now.getDay()));
      const lastWeekStart = new Date(new Date(startOfWeek).setDate(startOfWeek.getDate() - 7));

      const thisWeek = data.filter((entry) => entry.timestamp >= startOfWeek);
      const lastWeek = data.filter((entry) => entry.timestamp < startOfWeek && entry.timestamp >= lastWeekStart);

      const thisWeekAvg = thisWeek.length > 0
        ? thisWeek.reduce((sum, entry) => sum + entry.happiness, 0) / thisWeek.length
        : 0;
      const lastWeekAvg = lastWeek.length > 0
        ? lastWeek.reduce((sum, entry) => sum + entry.happiness, 0) / lastWeek.length
        : 0;

      setHappinessTrend(
        data.map((entry) => ({
          x: entry.timestamp.toISOString().split('T')[0],
          y: entry.happiness,
        }))
      );
      setAverageThisWeek(thisWeekAvg.toFixed(1));
      setChangeFromLastWeek(lastWeekAvg ? ((thisWeekAvg - lastWeekAvg) / lastWeekAvg * 100).toFixed(1) : null);
    };

    fetchStreak();
    fetchEntries();
  }, []);

  const generateReport = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId: auth.currentUser.uid }),
      });
      const data = await response.json();
      setReport(data);
    } catch (error) {
      console.error('Error generating report:', error);
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  const lineData = {
    datasets: [
      {
        label: 'Happiness Over Time',
        data: happinessTrend,
        parsing: {
          xAxisKey: 'x',
          yAxisKey: 'y',
        },
        borderColor: '#4CAF50',
        backgroundColor: 'rgba(76, 175, 80, 0.2)',
        fill: true,
      },
    ],
  };

  const donutData = {
    labels: ['This Week\'s Average Happiness'],
    datasets: [
      {
        data: [averageThisWeek, 10 - averageThisWeek],
        backgroundColor: ['#4CAF50', '#E0E0E0'],
      },
    ],
  };

  const lineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
        },
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Happiness Score',
        },
        min: 0,
        max: 10,
      },
    },
  };

  return (
    <div className="p-4 bg-gray-50 h-screen overflow-y-auto">
      <h1 className="text-2xl font-bold mb-2 text-center">
        Your Dashboard
      </h1>
      {streak > 0 && (
        <p className="text-lg text-green-600 text-center mb-4">
          Current Streak: {streak} day{streak > 1 ? 's' : ''}
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-lg font-semibold text-center mb-4">Average Happiness (This Week)</h2>
          <div className="w-full mx-auto">
            <Doughnut data={donutData} />
          </div>
          <div className="mt-4 bg-gray-100 p-4 rounded-lg shadow-md text-center">
            {changeFromLastWeek !== null ? (
              <p>
                Your happiness score is{' '}
                <span className={changeFromLastWeek > 0 ? 'text-green-600' : 'text-red-600'}>
                  {changeFromLastWeek > 0 ? 'up' : 'down'} {Math.abs(changeFromLastWeek)}%
                </span>{' '}
                compared to last week.
              </p>
            ) : (
              <p>No data available for last week to compare.</p>
            )}
          </div>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-center mb-4">Happiness Trend</h2>
          <div style={{ height: '200px' }}>
            <Line data={lineData} options={lineOptions} />
          </div>
        </div>
      </div>

      <div className="mt-8 text-center">
        <button
          onClick={generateReport}
          className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? 'Generating Report...' : 'Generate Mental Health Report'}
        </button>
        {report && (
          <div className="mt-4 bg-gray-100 p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold">Mental Health Report</h2>
            <p><strong>Brain Health Index:</strong> {report.brainHealthIndex}%</p>
            <p><strong>Mood Stability:</strong> {report.moodStability}%</p>
            
            <div className="mt-4">
              <p><strong>Anxiety Risk:</strong> {report.anxiety}%</p>
              {report.strengths.includes("Low risk of Anxiety.") && (
                <p className="text-green-600">Low risk of Anxiety.</p>
              )}
            </div>

            <div className="mt-4">
              <p><strong>Depression Risk:</strong> {report.depression}%</p>
              {report.strengths.includes("Low risk of Depression.") && (
                <p className="text-green-600">Low risk of Depression.</p>
              )}
            </div>

            <div className="mt-4">
              <p><strong>Psychosis Risk:</strong> {report.psychosis}%</p>
              {report.strengths.includes("Low risk of Psychosis.") && (
                <p className="text-green-600">Low risk of Psychosis.</p>
              )}
            </div>

            <div className="mt-4">
              <p><strong>Alzheimer's Risk:</strong> {report.alzheimers}%</p>
              {report.strengths.includes("Low risk of Alzheimers.") && (
                <p className="text-green-600">Low risk of Alzheimer's.</p>
              )}
            </div>

            <div className="mt-4">
              <p><strong>Dementia Risk:</strong> {report.dementia}%</p>
              {report.strengths.includes("Low risk of Dementia.") && (
                <p className="text-green-600">Low risk of Dementia.</p>
              )}
            </div>

            {/* Trend Analysis */}
            <div className="mt-4">
              <h3 className="text-md font-semibold">Trend Analysis</h3>
              <TrendVisualizer trend={report.trendAnalysis} />
            </div>
          </div>
        )}
      </div>

      <h2 className="text-lg font-semibold mt-6 mb-4">Recent Entries</h2>
      <ul className="space-y-4">
        {entries.map((entry) => (
          <li key={entry.id} className="p-4 bg-white shadow rounded-lg">
            <p className="text-sm text-gray-600">{entry.timestamp.toLocaleString()}</p>
            <p className="mt-2">{entry.entry}</p>
            <p className="mt-2 font-bold">Happiness: {entry.happiness}</p>
            <p className="mt-2 text-gray-600">
              <strong>Sentiment:</strong> {entry.sentiment}
            </p>
            <p className="mt-2 text-gray-600">
              <strong>Themes:</strong> {entry.themes.length > 0 ? entry.themes.join(', ') : 'None'}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
