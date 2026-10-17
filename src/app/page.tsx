import type { Metadata } from 'next';
import PortalHome from '@/components/portal/PortalHome';

export const metadata: Metadata = {
  title: 'Verantyx — VeRantyx: PC VR from a Mac, on vxrt',
  description:
    'VeRantyx runs Windows SteamVR games such as Half-Life: Alyx on an Apple silicon Mac and streams them to Apple Vision Pro or ALVR headsets, on vxrt — a Windows runtime written from scratch, no Wine. Also Vera, Cleanroom and the Verantyx apps.',
  openGraph: {
    title: 'VeRantyx — PC VR from a Mac',
    description: 'Half-Life: Alyx at 60 fps on a MacBook Pro, in Apple Vision Pro, on a Windows runtime written from scratch.',
    images: ['/media/verantyx/vxrt-frame.jpg'],
  },
};

export default function Home() {
  return <PortalHome />;
}
