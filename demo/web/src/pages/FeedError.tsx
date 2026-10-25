export function FeedError({ message }: { message: string }) {
  return (
    <p role="alert" className="px-3 py-6 text-center text-sm text-red-400">
      Could not load the feed: {message}
    </p>
  )
}
