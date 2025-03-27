"use client";

import { useEffect, useState } from "react";
import { SightingCard, type Sighting } from "../components/SightingCard";

// Refresh interval in milliseconds (e.g., 10 seconds)
const REFRESH_INTERVAL = 10000;

export default function Home() {
  const [sightings, setSightings] = useState<Sighting[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  async function loadSightings() {
    try {
      // Fetch sightings with a cache-busting query parameter
      const timestamp = new Date().getTime();
      const response = await fetch(`/sightings.json?t=${timestamp}`, {
        cache: 'no-store'
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch sightings');
      }
      
      const data = await response.json();
      setSightings(data);
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Error loading sightings:', error);
    } finally {
      setLoading(false);
    }
  }

  // Handle hiding a sighting from the UI immediately
  const handleHideSighting = (id: string) => {
    setSightings(prevSightings => 
      prevSightings.map(sighting => 
        sighting.id === id 
          ? { ...sighting, hidden: true } 
          : sighting
      )
    );
  };

  useEffect(() => {
    // Initial load
    loadSightings();
    
    // Set up auto-refresh interval
    const intervalId = setInterval(() => {
      loadSightings();
    }, REFRESH_INTERVAL);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Format the last refresh time
  const formattedLastRefresh = lastRefresh.toLocaleTimeString();

  // Filter out hidden sightings
  const visibleSightings = sightings.filter(sighting => !sighting.hidden);

  return (
    <section className="flex flex-col items-center justify-center w-full py-12 md:py-24 md:container md:mx-auto px-2 md:px-0 ">
      {loading ? (
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          <span className="ml-2">Loading sightings...</span>
        </div>
      ) : (
        <>
          <div className="w-full max-w-3xl mb-4 flex justify-between items-center  ">
            <h1 className="text-2xl font-bold">Bird Sightings</h1>
            <div className="text-sm text-gray-500">
              <span>Updated: {formattedLastRefresh}</span>
              <button 
                onClick={loadSightings} 
                className="ml-2 px-2 py-1 bg-tangerine text-white rounded text-xs hover:bg-orange-800"
              >
                Refresh
              </button>
            </div>
          </div>
          
          {visibleSightings.length === 0 ? (
            <div className="text-center py-10">
              <p>No bird sightings recorded yet</p>
            </div>
          ) : (
            <div className="max-w-3xl space-y-8 w-full ">
              {visibleSightings.map((sighting) => (
                <SightingCard 
                  key={sighting.id} 
                  sighting={sighting} 
                  onHide={handleHideSighting}
                />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
