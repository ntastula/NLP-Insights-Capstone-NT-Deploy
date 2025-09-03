import { useEffect } from "react";

function useUnloadCleanup() {
  useEffect(() => {
    const handleUnload = () => {
      // sendBeacon ensures the request is sent even when tab is closing
      const url = "/api/clear_user_data/";
      navigator.sendBeacon(url);
    };

    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, []);
}

export default useUnloadCleanup;
