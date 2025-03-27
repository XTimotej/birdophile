import { revalidatePath } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Extract the secret from the request headers
    const authHeader = request.headers.get('authorization');
    
    // Simple secret verification (should match what we'll set in Python)
    const expectedSecret = 'bird-camera-webhook-secret';
    const providedSecret = authHeader?.split(' ')[1] || '';
    
    if (providedSecret !== expectedSecret) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // Revalidate the homepage to refresh the data
    revalidatePath('/');
    
    return NextResponse.json({ success: true, revalidated: true });
  } catch (error) {
    return NextResponse.json({ error: 'Error revalidating' }, { status: 500 });
  }
} 