import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(request: Request) {
  try {
    const { id } = await request.json();
    
    if (!id) {
      return NextResponse.json(
        { error: 'Missing sighting ID' },
        { status: 400 }
      );
    }
    
    // Read the current sightings file
    const sightingsPath = path.join(process.cwd(), 'public', 'sightings.json');
    const sightingsData = fs.readFileSync(sightingsPath, 'utf8');
    const sightings = JSON.parse(sightingsData);
    
    // Find the sighting and mark it as hidden
    const updatedSightings = sightings.map((sighting: any) => {
      if (sighting.id === id) {
        return { ...sighting, hidden: true };
      }
      return sighting;
    });
    
    // Write the updated sightings back to the file
    fs.writeFileSync(sightingsPath, JSON.stringify(updatedSightings, null, 2));
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error hiding sighting:', error);
    return NextResponse.json(
      { error: 'Failed to hide sighting' },
      { status: 500 }
    );
  }
} 