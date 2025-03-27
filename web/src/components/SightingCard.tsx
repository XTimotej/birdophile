"use client";

import { useState } from "react";
import Image from "next/image";

export type Sighting = {
  id: string;
  timestamp: string;
  date: string;
  time: string;
  image: string;
  video: string;
  type: string;
  hidden?: boolean;
};

export function SightingCard({ sighting, onHide }: { sighting: Sighting, onHide?: (id: string) => void }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isHiding, setIsHiding] = useState(false);
  
  const handleHide = async () => {
    if (isHiding) return;
    
    try {
      setIsHiding(true);
      const response = await fetch('/api/hide-sighting', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id: sighting.id }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to hide sighting');
      }
      
      // If onHide callback is provided, call it to update the parent component
      if (onHide) {
        onHide(sighting.id);
      }
    } catch (error) {
      console.error('Error hiding sighting:', error);
      alert('Failed to hide this sighting. Please try again.');
    } finally {
      setIsHiding(false);
    }
  };
  
  return (
    <div className="bg-tangerine rounded-2xl shadow-md overflow-hidden">
      <div className="relative group overflow-hidden">
        {!isPlaying ? (
          <>
            <Image 
              src={`/images/${sighting.image}`} 
              alt={`Sighting ${sighting.id}`}
              width={1920}  
              height={1080}
              className="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-110"
            />
            <button 
              onClick={() => setIsPlaying(true)}
              className="absolute inset-0 flex items-center justify-center bg-black/30 hover:bg-black/10 hover:cursor-pointer transition-all"
            >
              <div className="w-32 h-32 flex items-center justify-center opacity-20">
                <svg viewBox="-1.315 -1.315 42 42" xmlns="http://www.w3.org/2000/svg" height="96" width="96">
                  <path d="M1.2303125 19.681719166666664a18.4546875 18.4546875 0 1 0 36.909375 0 18.4546875 18.4546875 0 1 0 -36.909375 0Z" fill="none" stroke="#ffffff" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1"></path>
                  <path d="M14.76375 25.611825416666665a2.6837216666666666 2.6837216666666666 0 0 0 4.44881 2.0193529166666666L28.2971875 19.685l-9.0846275 -7.951099583333334A2.68208125 2.68208125 0 0 0 14.76375 13.753253333333333Z" fill="none" stroke="#ffffff" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1"></path>
                </svg>
              </div>
            </button>
          </>
        ) : (
          <video 
            src={`/videos/${sighting.video}`} 
            className="w-full h-full object-cover"
            autoPlay 
            controls
            onEnded={() => setIsPlaying(false)}
          />
        )}
      </div>
      
      <div className="p-4">
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center gap-2">
            {/* {sighting.type} */}
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" height="24" width="24" className="w-5 h-5">
            <g ><path id="Union" fill="#000000" fillRule="evenodd" d="M14.9999 1h-1.5166L0.00439453 20H10v3h2v-3h0.9999c0.3381 0 0.6717 -0.0186 1.0001 -0.0549V23h2v-3.5121c3.4955 -1.2356 5.9999 -4.5693 5.9999 -8.4879V9H24V7h-2.0709c-0.4853 -3.39229 -3.4027 -6 -6.9292 -6ZM16 9V6h2v3h-2Z" clipRule="evenodd" strokeWidth="1"></path></g></svg>
            <span className="text-sm ">{sighting.date}</span>
          </div>
          <div className="flex items-center gap-2">
          <span className="text-sm ">{sighting.time}</span>
            <button
            onClick={handleHide}
            disabled={isHiding}
            className="px-2 py-1 bg-black text-white text-xs font-medium rounded hover:bg-red-700 transition-colors disabled:opacity-50"
          >
            {isHiding ? 'Hiding...' : 'Hide'}
          </button>
          </div>
          
        </div>

      </div>
    </div>
  );
} 