import { useEffect, useState } from 'react'
import { parseRoute, type Route } from './lib'

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash))
  useEffect(() => {
    const on = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}
export const navigate = (hash: string) => {
  window.location.hash = hash
}
