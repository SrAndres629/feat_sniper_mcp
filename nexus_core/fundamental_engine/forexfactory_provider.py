import pandas as pd
import requests
import datetime
from io import StringIO
from typing import List, Optional
from .calendar_client import EconomicEvent, EventImpact

class ForexFactoryProvider:
    """
    [MACRO SENTINEL - DATA PROVIDER]
    Fetches real economic data from ForexFactory CSV Export.
    """
    # [INSTITUTIONAL PROXY] Using Supabase Edge Function to bypass local 403/WAF blocks.
    CSV_URL = "https://hkbnbwcjbitadkdkgzco.supabase.co/functions/v1/macro-proxy"
    HEADERS = {
        "User-Agent": "FEAT-Sniper/1.0"
    }

    def fetch_events(self) -> List[EconomicEvent]:
        """
        Fetches and parses the CSV data from our Supabase Macro Proxy.
        Includes a robust HTML fallback if CSV is blocked.
        """
        try:
            response = requests.get(self.CSV_URL, headers=self.HEADERS, timeout=15)
            response.raise_for_status()
            
            # Case 1: Proxy succeeded in getting CSV
            if response.headers.get("X-Bypass-Status") == "SUCCESS" or "Date,Time" in response.text[:100]:
                df = pd.read_csv(
                    StringIO(response.text), 
                    on_bad_lines='skip',
                    engine='python',
                    skipinitialspace=True
                )
                return self._parse_dataframe(df)
            
            # Case 2: Proxy returned HTML (Bypass failed)
            print("⚠️ MACRO: CSV Blocked. Attempting HTML extraction fallback...")
            return self._parse_html_fallback(response.text)
            
        except Exception as e:
            print(f"⚠️ MACRO ERROR: Failed to fetch data: {e}")
            return []

    def _parse_html_fallback(self, html_content: str) -> List[EconomicEvent]:
        """
        [DOCTORAL EMERGENCY] Parses events directly from ForexFactory HTML.
        This is slower but extremely robust against CSV export blocks.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        events = []
        now = datetime.datetime.now()
        
        # ForexFactory table rows
        rows = soup.find_all("tr", class_="calendar__row")
        
        current_date = None
        for row in rows:
            try:
                # 1. Date (only present on first row of the day)
                # Classes: "calendar__cell calendar__date date"
                date_cell = row.find("td", class_=lambda x: x and "calendar__date" in x)
                if date_cell and date_cell.text.strip():
                    current_date = date_cell.text.strip() # e.g. "Sun Jan 18"
                
                if not current_date: continue
                
                # 2. Time
                time_cell = row.find("td", class_=lambda x: x and "calendar__time" in x)
                time_str = time_cell.text.strip() if time_cell else "All Day"
                
                # 3. Currency
                curr_cell = row.find("td", class_=lambda x: x and "calendar__currency" in x)
                currency = curr_cell.text.strip() if curr_cell else ""
                
                # 4. Impact (Usually an icon)
                impact = EventImpact.LOW
                impact_cell = row.find("td", class_=lambda x: x and "calendar__impact" in x)
                if impact_cell:
                    impact_span = impact_cell.find("span")
                    if impact_span:
                        cls_str = str(impact_span.get("class", [])).lower()
                        if "red" in cls_str: impact = EventImpact.HIGH
                        elif "ora" in cls_str: impact = EventImpact.MEDIUM
                        elif "yel" in cls_str: impact = EventImpact.LOW

                # 5. Event Name
                event_cell = row.find("td", class_=lambda x: x and "calendar__event" in x)
                event_name = event_cell.text.strip() if event_cell else ""
                
                if not event_name or not currency: 
                    continue

                # 6. Parse Timestamp
                year = now.year
                try:
                    # Date format: "Sun Jan 18" -> we need "Jan 18"
                    date_parts = current_date.split()
                    clean_date = " ".join(date_parts[1:3]) if len(date_parts) >= 3 else current_date
                    
                    if "Day" in time_str or "Tentative" in time_str:
                         timestamp = datetime.datetime.strptime(f"{clean_date} {year}", "%b %d %Y")
                    else:
                         # Normalize time: "4:00am"
                         clean_time = time_str.replace(" ", "").lower()
                         timestamp = datetime.datetime.strptime(f"{clean_date} {year} {clean_time}", "%b %d %Y %I:%M%p")
                except:
                    continue

                events.append(EconomicEvent(
                    timestamp=timestamp,
                    currency=currency,
                    event_name=event_name,
                    impact=impact
                ))
            except Exception:
                continue
                
        return sorted(events, key=lambda x: x.timestamp)

    def _parse_dataframe(self, df: pd.DataFrame) -> List[EconomicEvent]:
        """
        Normalizes ForexFactory CSV into FEAT EconomicEvent objects.
        CSV Structure: Date,Time,Currency,Impact,Event,Actual,Forecast,Previous
        """
        events = []
        now = datetime.datetime.now()
        
        for _, row in df.iterrows():
            try:
                # 1. Parse Timestamp (ForexFactory uses MM-DD-YYYY and HH:MMam/pm)
                # Note: This usually reflects the user's timezone on the site, 
                # but the CSV export is generally UTC or EST based on FF settings.
                # We assume the default FF export timezone for now.
                date_str = f"{row['Date']} {row['Time']}"
                
                # Check for "All Day" or Tentative events
                if "am" not in date_str.lower() and "pm" not in date_str.lower():
                    # Handle Day-long events as 00:00
                    timestamp = datetime.datetime.strptime(row['Date'], "%m-%d-%Y")
                else:
                    timestamp = datetime.datetime.strptime(date_str, "%m-%d-%Y %I:%M%p")

                # 2. Map Impact
                impact_map = {
                    "Low": EventImpact.LOW,
                    "Medium": EventImpact.MEDIUM,
                    "High": EventImpact.HIGH
                }
                impact = impact_map.get(row['Impact'], EventImpact.LOW)

                # 3. Create Event
                event = EconomicEvent(
                    timestamp=timestamp,
                    currency=row['Currency'],
                    event_name=row['Event'],
                    impact=impact,
                    forecast=self._clean_val(row.get('Forecast')),
                    previous=self._clean_val(row.get('Previous')),
                    actual=self._clean_val(row.get('Actual'))
                )
                
                # Only keep future events or very recent ones
                if timestamp > now - datetime.timedelta(hours=2):
                    events.append(event)
                    
            except Exception:
                continue
                
        return sorted(events, key=lambda x: x.timestamp)

    def _clean_val(self, val) -> Optional[float]:
        """Handles non-numeric values in CSV (e.g., '200K', '0.5%')."""
        if pd.isna(val) or val == "": return None
        try:
            # Remove %, K, M and convert to float
            s = str(val).replace("%", "").replace("K", "").replace("M", "").strip()
            return float(s)
        except ValueError:
            return None
