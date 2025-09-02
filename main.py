// Backend base URL (Render or local). You can override with env var BACKEND_BASE at build/run time.
import SwiftUI
import Foundation
import UIKit

let BACKEND_BASE: String = ProcessInfo.processInfo.environment["BACKEND_BASE"] ?? "https://smartpicks-backend.onrender.com"

// Point iOS -> your Render backend
// Dismiss keyboard helper (auto-dismiss + optional extra action)
extension View {
    func keyboardToolbarDone(_ extra: (() -> Void)? = nil) -> some View {
        self.toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                    to: nil, from: nil, for: nil)
                    extra?()
                }
            }
        }
    }
    /// Tap anywhere in this view to dismiss the keyboard.
    func hideKeyboardOnTap() -> some View {
        self.onTapGesture {
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                            to: nil, from: nil, for: nil)
        }
    }
}
// MARK: - Odds Model
struct Odd: Codable, Identifiable, Hashable {
    var id: String { gameId + ":" + market + ":" + selection }
    let gameId: String
    let league: String
    let market: String
    let selection: String
    let price: Double
    let source: String?
}
// MARK: - Sport → League Mapping (Backend Codes)
extension Sport {
    var leagueCode: String {
        switch self {
        case .NBA: return "nba"
        case .NFL: return "nfl"
        case .CFB: return "ncaa_football"   // important: many backends use this (not "cfb")
        case .MLB: return "mlb"
        case .NHL: return "nhl"
        }
    }
}

// Format the EXACT day from the user's calendar selection (no TZ drift)
fileprivate func ymdLocal(from date: Date) -> String {
    let cal = Calendar.current
    let comps = cal.dateComponents([.year, .month, .day], from: date)
    let y = comps.year ?? 1970
    let m = comps.month ?? 1
    let d = comps.day ?? 1
    return String(format: "%04d-%02d-%02d", y, m, d)
}
// MARK: - Odds Networking (with Full Logging)
func fetchOddsWithLogging(
    leagueCode: String,
    date: Date,
    market: String,        // "player_props" or "moneyline"
    allUpcoming: Bool,
    setError: @escaping (String?) -> Void,
    setOdds: @escaping ([Odd]) -> Void
) {
    let df = DateFormatter()
    df.dateFormat = "yyyy-MM-dd"
    let dateStr = df.string(from: date)

    let route = allUpcoming ? "/api/upcoming" : "/api/odds"
    var comps = URLComponents(string: BACKEND_BASE + route)!
    comps.queryItems = [
        URLQueryItem(name: "league", value: leagueCode),
        URLQueryItem(name: "date", value: dateStr),
        URLQueryItem(name: "market", value: market)
    ]
    guard let url = comps.url else { setError("Bad URL"); return }

    var req = URLRequest(url: url)
    req.httpMethod = "GET"
    req.setValue("application/json", forHTTPHeaderField: "Accept")
    if !ODDS_API_KEY.isEmpty {
        req.setValue("Bearer \(ODDS_API_KEY)", forHTTPHeaderField: "Authorization")
    }

    // LOG request clearly
    print("🔵 REQUEST URL:\n\(url.absoluteString)")
    print("🔵 REQUEST cURL:\n\(req.asCurl)")

    URLSession.shared.dataTask(with: req) { data, resp, err in
        if let err = err {
            let msg = "Network error: \(err.localizedDescription)"
            print("🔴 \(msg)")
            DispatchQueue.main.async { setError(msg); setOdds([]) }
            return
        }
        guard let http = resp as? HTTPURLResponse else {
            let msg = "No HTTP response"
            print("🔴 \(msg)")
            DispatchQueue.main.async { setError(msg); setOdds([]) }
            return
        }

        print("🟣 STATUS: \(http.statusCode)")
        print("🟣 HEADERS:\n\(http.allHeaderFields.asHTTPHeaderLines)")
        let body = data ?? Data()
        let bodyText = prettyJSON(body)
        print("🟣 RAW BODY (\(body.count) bytes):\n\(bodyText)")

        guard (200...299).contains(http.statusCode) else {
            // show full backend error (e.g., 404/422 with {"detail": ...})
            let detail: String
            if let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
               let d = obj["detail"] as? String {
                detail = d
            } else {
                detail = bodyText.isEmpty ? "HTTP \(http.statusCode)" : bodyText
            }
            let msg = "Backend \(http.statusCode): \(detail)"
            print("🟠 \(msg)")
            DispatchQueue.main.async { setError(msg); setOdds([]) }
            return
        }

        do {
            let decoded = try JSONDecoder().decode([Odd].self, from: body)
            DispatchQueue.main.async { setError(nil); setOdds(decoded) }
        } catch {
            let msg = "Decode error: \(error.localizedDescription)"
            print("🟠 \(msg)")
            DispatchQueue.main.async { setError(msg); setOdds([]) }
        }
    }.resume()
}
// MARK: - Debug Helpers (Request/Response Logging)
fileprivate extension Dictionary where Key == AnyHashable, Value == Any {
    var asHTTPHeaderLines: String {
        self.compactMap { "\($0.key): \($0.value)" }.sorted().joined(separator: "\n")
    }
}
fileprivate extension URLRequest {
    var asCurl: String {
        var parts = ["curl -i", "-X \(self.httpMethod ?? "GET")"]
        if let headers = self.allHTTPHeaderFields {
            for (k, v) in headers { parts.append("-H '\(k): \(v)'") }
        }
        if let body = self.httpBody, let s = String(data: body, encoding: .utf8), !s.isEmpty {
            parts.append("--data '\(s.replacingOccurrences(of: "'", with: "\\'"))'")
        }
        parts.append("'\(self.url?.absoluteString ?? "")'")
        return parts.joined(separator: " ")
    }
}
fileprivate func prettyJSON(_ data: Data) -> String {
    if let obj = try? JSONSerialization.jsonObject(with: data),
       let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]),
       let s = String(data: pretty, encoding: .utf8) { return s }
    return String(data: data, encoding: .utf8) ?? "<non-utf8 \(data.count) bytes>"
}
// MARK: - 🔑 Configure your API Key
private let ODDS_API_KEY = "58fb5f2de1ba933b00223293e99e740b" // demo/testing only

// MARK: - Sport Configuration

enum Sport: String, CaseIterable, Identifiable {
    case NBA, NFL, CFB, MLB, NHL
    var id: String { rawValue }

    var apiKey: String {
        switch self {
        case .NBA: return "basketball_nba"
        case .NFL: return "americanfootball_nfl"
        case .CFB: return "americanfootball_ncaaf"
        case .MLB: return "baseball_mlb"
        case .NHL: return "icehockey_nhl"
        }
    }
}

// MARK: - Dynamic MarketKey (accept ANY key returned by the API)

struct MarketKey: Hashable, Identifiable {
    let key: String
    var id: String { key }
    var label: String {
        switch key {
        // Basketball
        case "player_points": return "Points"
        case "player_rebounds": return "Rebounds"
        case "player_assists": return "Assists"
        case "player_threes": return "3PM"
        case "player_steals": return "Steals"
        case "player_blocks": return "Blocks"
        case "player_turnovers": return "TOs"
        case "player_points_rebounds_assists", "player_pra": return "PRA"
        case "player_points_rebounds", "player_pr": return "P+R"
        case "player_rebounds_assists", "player_ra": return "R+A"
        case "player_points_assists", "player_pa": return "P+A"
        // Football
        case "player_pass_yards": return "Pass Yds"
        case "player_pass_attempts": return "Pass Att"
        case "player_pass_completions": return "Comp"
        case "player_pass_tds": return "Pass TD"
        case "player_interceptions": return "INTs"
        case "player_rush_yards": return "Rush Yds"
        case "player_rush_attempts": return "Rush Att"
        case "player_rush_tds": return "Rush TD"
        case "player_receiving_yards": return "Rec Yds"
        case "player_receptions": return "Receptions"
        case "player_receiving_tds": return "Rec TD"
        case "player_longest_reception": return "Long Rec"
        case "player_longest_rush": return "Long Rush"
        // Baseball
        case "player_hits": return "Hits"
        case "player_runs": return "Runs"
        case "player_rbis": return "RBIs"
        case "player_home_runs": return "HR"
        case "player_total_bases": return "TB"
        case "player_walks": return "BB"
        case "player_strikeouts": return "K"
        case "pitcher_strikeouts": return "K (P)"
        case "pitcher_outs": return "Outs (P)"
        case "pitcher_hits_allowed": return "HA (P)"
        // Hockey
        case "player_shots_on_goal": return "SOG"
        case "player_goals": return "Goals"
        case "player_assists": return "Assists"
        case "player_points": return "Points"
        case "goalie_saves": return "Saves"
        default:
            return key.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}

// MARK: - Correlation rules (top-level so they don't initialize on the MainActor)
fileprivate let CORRELATION_RULES: [String: [(other: String, bonus: Double, reason: String)]] = [
    // --- NFL / CFB ---
    "player_pass_yards": [
        ("player_receiving_yards", 0.18, "QB ⇄ WR/TE passing-to-receiving yardage"),
        ("player_receptions",      0.16, "QB volume ↔ receiver catches"),
        ("player_pass_completions",0.12, "More completions → more yards"),
        ("player_pass_attempts",   0.08, "Attempts ↔ passing volume"),
        ("player_pass_tds",        0.06, "Yardage can lead to TDs")
    ],
    "player_pass_completions": [
        ("player_pass_yards",      0.12, "Completions → passing yards"),
        ("player_receptions",      0.10, "Team completions ↔ receiver catches")
    ],
    "player_pass_attempts": [
        ("player_pass_yards",      0.08, "Attempts ↔ passing volume"),
        ("player_pass_completions",0.10, "Attempts → completions")
    ],
    "player_pass_tds": [
        ("player_receiving_tds",   0.16, "QB TDs ↔ receiver TDs"),
        ("player_pass_yards",      0.06, "Yardage sometimes leads to TDs")
    ],
    "player_receiving_yards": [
        ("player_receptions",      0.14, "Catches drive receiving yards"),
        ("player_longest_reception",0.10, "Big plays ↔ yard totals"),
        ("player_pass_yards",      0.18, "Receiver tied to QB passing volume"),
        ("player_receiving_tds",   0.08, "Yardage can translate to TDs")
    ],
    "player_receptions": [
        ("player_receiving_yards", 0.14, "Catches drive receiving yards"),
        ("player_pass_yards",      0.16, "Receiver tied to QB passing volume")
    ],
    "player_receiving_tds": [
        ("player_pass_tds",        0.16, "Receiver TDs ↔ QB TDs"),
        ("player_receiving_yards", 0.08, "Yardage often accompanies TDs")
    ],
    "player_longest_reception": [
        ("player_receiving_yards", 0.10, "Explosive plays boost yard totals")
    ],
    "player_rush_attempts": [
        ("player_rush_yards",      0.12, "Volume ↔ rushing yards"),
        ("player_rush_tds",        0.08, "Attempts near goal line ↔ TDs")
    ],
    "player_rush_yards": [
        ("player_rush_attempts",   0.12, "Volume ↔ rushing yards"),
        ("player_rush_tds",        0.10, "Ground success ↔ rushing TDs"),
        ("player_longest_rush",    0.08, "Explosive rushes ↔ yard totals")
    ],
    "player_longest_rush": [
        ("player_rush_yards",      0.08, "Explosive rushes boost yard totals")
    ],
    "player_rush_tds": [
        ("player_rush_yards",      0.10, "Ground success ↔ rushing TDs")
    ],

    // --- NBA ---
    "player_points": [
        ("player_assists",         0.10, "Facilitator feeding scorer"),
        ("player_threes",          0.08, "Points ↔ 3PT makes"),
        ("player_points_rebounds_assists", 0.06, "Scoring contributes to PRA")
    ],
    "player_assists": [
        ("player_points",          0.10, "Assists ↔ scorer points"),
        ("player_threes",          0.06, "Assists leading to 3PT makes")
    ],
    "player_threes": [
        ("player_points",          0.08, "3PT makes contribute to points"),
        ("player_assists",         0.06, "Catch-and-shoot assisted threes")
    ],
    "player_rebounds": [
        ("player_points_rebounds_assists", 0.10, "Boards add to PRA"),
        ("player_points_rebounds",         0.10, "Boards add to P+R")
    ],
    "player_points_rebounds_assists": [
        ("player_points",          0.06, "PRA includes points"),
        ("player_rebounds",        0.10, "PRA includes rebounds"),
        ("player_assists",         0.08, "PRA includes assists")
    ],
    "player_points_assists": [
        ("player_points",          0.08, "Includes points"),
        ("player_assists",         0.08, "Includes assists")
    ],
    "player_points_rebounds": [
        ("player_points",          0.08, "Includes points"),
        ("player_rebounds",        0.10, "Includes rebounds")
    ],
    "player_rebounds_assists": [
        ("player_rebounds",        0.08, "Includes rebounds"),
        ("player_assists",         0.08, "Includes assists")
    ],
    "player_steals": [ ("player_blocks", 0.04, "Defensive activity cluster") ],
    "player_blocks": [ ("player_rebounds", 0.05, "Rim protection ↔ boards"), ("player_steals", 0.04, "Defensive activity cluster") ],

    // --- MLB ---
    "player_hits": [ ("player_total_bases", 0.12, "Hits contribute to total bases"), ("player_runs", 0.06,  "Getting on base → runs") ],
    "player_total_bases": [ ("player_hits", 0.12, "Singles/doubles/triples/HRs → TB"), ("player_home_runs", 0.10, "HRs drive TB") ],
    "player_home_runs": [ ("player_total_bases", 0.10, "HRs drive TB"), ("player_rbis", 0.06,  "HRs bring in RBIs"), ("player_runs", 0.06,  "HR always scores a run") ],
    "player_walks": [ ("player_runs", 0.05,  "Reaching base ↔ runs") ],
    "player_rbis": [ ("player_hits", 0.06,  "Hits with men on base ↔ RBIs") ],
    "pitcher_strikeouts": [ ("pitcher_outs", 0.10, "Deep outings ↔ Ks"), ("pitcher_hits_allowed", 0.05, "Weak contact ↔ more Ks (light)") ],
    "pitcher_outs": [ ("pitcher_strikeouts", 0.10, "Deep outings ↔ Ks") ],

    // --- NHL ---
    "player_shots_on_goal": [ ("player_points", 0.08, "Shot volume ↔ points"), ("player_goals", 0.10, "Shot volume ↔ goals") ],
    "player_goals": [ ("player_points", 0.10, "Goals are points"), ("player_shots_on_goal", 0.10, "More shots → more goals") ],
    "player_assists": [ ("player_points", 0.08, "Assists are points") ],
    "player_points": [ ("player_goals", 0.10, "Goals are points"), ("player_assists", 0.08, "Assists are points"), ("player_shots_on_goal", 0.08, "Shot volume ↔ points") ],

    // --- Team markets (generic synergies) ---
    "totals": [
        ("player_points",  0.05, "High-scoring game favors scorers/overs"),
        ("player_assists", 0.03, "High total lifts assist chances"),
        ("player_threes",  0.03, "High total lifts 3PT volume")
    ],
    "h2h": [
        ("spreads",        0.04, "Favorites often correlate with spread cover"),
        ("player_points",  0.03, "Favorites lean on star scoring")
    ],
    "spreads": [
        ("h2h",            0.04, "Covering spread aligns with winning SU"),
        ("player_points",  0.03, "Covering spread aligns with star usage")
    ],
]


enum Tier: String, CaseIterable, Identifiable { case Safe, Medium, Risky, Lotto; var id: String { rawValue } }
enum RiskMode: String, CaseIterable, Identifiable { case Conservative, Moderate, Aggressive; var id: String { rawValue } }

struct Pick: Identifiable, Hashable {
    let id = UUID()
    let sport: Sport
    let eventId: String
    let eventLabel: String // AWAY @ HOME
    let commence: Date
    let player: String
    let marketKey: MarketKey
    let line: Double?
    let overOdds: Double           // American odds (Over or only outcome)
    let underOdds: Double?         // American odds (Under if available)
    let bookmaker: String
    let confidence: Double         // 0...1
    let tier: Tier
    let badges: [String]
    var correlationGroup: String? { eventId }
}

// MARK: - Odds API DTOs

struct APIEvent: Decodable {
    let id: String
    let sport_key: String
    let sport_title: String
    let commence_time: String
    let home_team: String
    let away_team: String
    let bookmakers: [APIBookmaker]
}

struct APIBookmaker: Decodable {
    let key: String
    let title: String
    let last_update: String
    let markets: [APIMarket]
}

struct APIMarket: Decodable {
    let key: String
    let outcomes: [APIOutcome]
}

struct APIOutcome: Decodable {
    // Some books: name = "Over"/"Under", description = Player
    // Others: name = Player
    let name: String
    let price: Double
    let point: Double?
    let description: String?

    var playerName: String {
        let lower = name.lowercased()
        if lower == "over" || lower == "under" { return description ?? "Unknown" }
        return name
    }
    var isOver: Bool? {
        let lower = name.lowercased()
        if lower == "over" { return true }
        if lower == "under" { return false }
        return nil
    }
}

@MainActor
final class OddsAPIClient: ObservableObject {
    static let shared = OddsAPIClient()
    
    @Published var requestsRemaining: Int? = nil
    @Published var lastUpdated: Date? = nil
    
    private let session: URLSession
    private var cache: [String: (stamp: Date, events: [APIEvent])] = [:]
    private let ttl: TimeInterval = 10 * 60 // cache 10 mins
    
    private init() {
        let cfg = URLSessionConfiguration.default
        cfg.waitsForConnectivity = true
        cfg.timeoutIntervalForRequest = 25
        cfg.httpAdditionalHeaders = ["Accept": "application/json"]
        session = URLSession(configuration: cfg)
    }
    
    /// Fetch all props for a sport from your backend (optionally scoped to a calendar day).
    func fetchAllProps(sport: Sport, date: Date? = nil, markets: [String]? = nil) async throws -> [APIEvent] {
        // Use local calendar day (no TZ drift)
        let dayStr: String? = date.map { ymdLocal(from: $0) }
        // Fallback to previous local day (some backends key by UTC)
        let fallbackDayStr: String? = {
            guard let d = date else { return nil }
            return Calendar.current.date(byAdding: .day, value: -1, to: d).map(ymdLocal(from:))
        }()

        // cache key
        let key = [
            sport.rawValue,
            dayStr ?? "upcoming",
            (markets ?? []).sorted().joined(separator: ",")
        ].joined(separator: "|")

        if let c = cache[key], Date().timeIntervalSince(c.stamp) < ttl {
            return c.events
        }

        // Candidates to try
        let routes = (dayStr == nil) ? ["/api/upcoming", "/upcoming"] : ["/api/odds", "/odds"]
        let paramNames = ["league", "sport"]                // some servers want league=, others sport=
        let sportValues = [sport.leagueCode, sport.apiKey]  // e.g. "mlb" and "baseball_mlb"
        let marketParam = (markets?.isEmpty == false ? markets!.joined(separator: ",") : "all")
        let datesToTry: [String?] = (dayStr == nil) ? [nil] : [dayStr, fallbackDayStr].compactMap { $0 }

        var lastErrorBody = "(no body)"
        var lastStatus = -1
        var lastURL: URL?

        for route in routes {
            for pname in paramNames {
                for code in sportValues {
                    for dstr in datesToTry {
                        var comps = URLComponents(string: "\(BACKEND_BASE)\(route)")!
                        var items: [URLQueryItem] = [
                            URLQueryItem(name: pname, value: code),
                            URLQueryItem(name: "market", value: marketParam)
                        ]
                        if let d = dstr { items.append(URLQueryItem(name: "date", value: d)) }
                        comps.queryItems = items
                        guard let url = comps.url else { continue }
                        lastURL = url

                        var req = URLRequest(url: url)
                        req.httpMethod = "GET"
                        req.setValue("application/json", forHTTPHeaderField: "Accept")
                        if !ODDS_API_KEY.isEmpty {
                            req.setValue("Bearer \(ODDS_API_KEY)", forHTTPHeaderField: "Authorization")
                        }

                        print("🔵 REQUEST: \(req.httpMethod ?? "GET") \(url.absoluteString)")

                        let data: Data
                        let resp: URLResponse
                        do {
                            (data, resp) = try await session.data(for: req)
                        } catch {
                            lastStatus = -1
                            lastErrorBody = error.localizedDescription
                            print("🔴 REQUEST ERROR: \(error.localizedDescription)")
                            continue
                        }

                        guard let http = resp as? HTTPURLResponse else {
                            lastStatus = -1
                            lastErrorBody = "No HTTP response"
                            continue
                        }

                        lastStatus = http.statusCode
                        let bodyText = String(data: data, encoding: .utf8) ?? "(binary \(data.count) bytes)"
                        print("🟣 STATUS: \(http.statusCode) for \(route) with \(pname)=\(code) date=\(dstr ?? "nil")")
                        print("🟣 BODY:\n\(bodyText)")

                        if let rem = http.value(forHTTPHeaderField: "x-requests-remaining"),
                           let n = Int(rem) { self.requestsRemaining = n }
                        self.lastUpdated = Date()

                        if (200...299).contains(http.statusCode) {
                            do {
                                let events = try JSONDecoder().decode([APIEvent].self, from: data)
                                cache[key] = (Date(), events)
                                return events
                            } catch {
                                throw URLError(.cannotParseResponse,
                                               userInfo: ["message": "Decoding failed: \(bodyText)"])
                            }
                        }

                        lastErrorBody = bodyText
                    }
                }
            }
        }

        let urlStr = lastURL?.absoluteString ?? "\(BACKEND_BASE)(no-url)"
        throw URLError(.badServerResponse,
                       userInfo: ["message": "Backend \(lastStatus) at \(urlStr): \(lastErrorBody)"])
    }
    }// MARK: - View Model (live data + logic)

    @MainActor
    final class AppVM: ObservableObject {
        // State
        @Published var selectedSport: Sport = .NBA
        @Published var date: Date = Date()
        @Published var showAllUpcoming: Bool = false
        
        // Settings (persisted)
        @AppStorage("bankroll") var bankroll: Double = 1000
        @AppStorage("riskMode") private var riskModeRaw: String = RiskMode.Moderate.rawValue
        var riskMode: RiskMode { get { RiskMode(rawValue: riskModeRaw) ?? .Moderate } set { riskModeRaw = newValue.rawValue; objectWillChange.send() } }
        @AppStorage("dailyLossCapEnabled") var dailyLossCapEnabled: Bool = true
        @AppStorage("dailyLossCapAmount") var dailyLossCapAmount: Double = 100
        @AppStorage("marketParityMode") var marketParityMode: Bool = false
        @AppStorage("lastSport") private var lastSportRaw: String = Sport.NBA.rawValue
        // Correlation rules are defined top-level in `CORRELATION_RULES` (see above); no per-instance copies here.

        // Hedge persistence
        @AppStorage("hedgePositionsJSON") private var hedgeJSON: String = "[]"

        // Data
        @Published var topPicks: [Pick] = []
        @Published var builderLegs: [Pick] = []

        // Loading / Errors
        @Published var isLoading = false
        @Published var errorMessage: String? = nil

        private let api = OddsAPIClient.shared

        init() {
            if let s = Sport(rawValue: lastSportRaw) { selectedSport = s }
        }

        func refresh() async {
            isLoading = true
            defer { isLoading = false }
            errorMessage = nil
            do {
                let events = try await api.fetchAllProps(
                    sport: selectedSport,
                    date: showAllUpcoming ? nil : date
                )
                topPicks = makePicks(from: events, sport: selectedSport)
                lastSportRaw = selectedSport.rawValue
                let eventIds = events.map { $0.id }
                await fetchSignalsIfAvailable(for: eventIds)
                // If backend has no signals, derive lightweight local ones so correlation UI has data.
                if signalsByEventId.isEmpty {
                    buildDerivedSignals()
                }
            } catch {
                let msg = (error as NSError).userInfo["message"] as? String ?? error.localizedDescription
                errorMessage = "Odds fetch failed: \(msg)"
                topPicks = []
            }
        }

        // Pair Over/Under by (player, market, line) within preferred bookmaker; fallback across books
        private func makePicks(from events: [APIEvent], sport: Sport) -> [Pick] {
            var picks: [Pick] = []
            let df = ISO8601DateFormatter()
            let cal = Calendar.current

            for ev in events {
                guard let commence = df.date(from: ev.commence_time) else { continue }
                if !showAllUpcoming {
                    if !cal.isDate(commence, inSameDayAs: date) { continue }
                }
                let label = "\(ev.away_team) @ \(ev.home_team)"

                // Sort books: FanDuel first
                let books = ev.bookmakers.sorted { a, b in
                    let ak = a.key.lowercased(); let bk = b.key.lowercased()
                    if ak == "fanduel" { return true }
                    if bk == "fanduel" { return false }
                    return a.title < b.title
                }

                var seenCombos = Set<String>() // event|market|player|line to avoid duplicates across books
                for book in books {
                    for m in book.markets {
                        let mk = MarketKey(key: m.key)
                        // --- Team markets (Moneyline / Spread) special-cases ---
                        if mk.key == "h2h" {
                            // Moneyline: create one pick per team; no line; price is overOdds
                            for out in m.outcomes {
                                let team = out.name
                                let dedupeKey = "\(ev.id)|\(mk.key)|\(team)|-9999"
                                if seenCombos.contains(dedupeKey) { continue }
                                seenCombos.insert(dedupeKey)

                                let oPrice = out.price
                                let implied = impliedProb(oPrice)
                                let conf = max(0.35, min(0.9,
                                                         0.5 + (abs(oPrice) >= 140 ? 0.18 : 0.0) + (implied > 0.55 ? 0.07 : -0.03)))
                                let tier = tier(for: conf)
                                var badges = [book.title]

                                picks.append(Pick(
                                    sport: sport,
                                    eventId: ev.id,
                                    eventLabel: label,
                                    commence: commence,
                                    player: team,                  // show team name in player field
                                    marketKey: mk,
                                    line: nil,
                                    overOdds: oPrice,              // single-sided
                                    underOdds: nil,
                                    bookmaker: book.title,
                                    confidence: conf,
                                    tier: tier,
                                    badges: badges
                                ))
                            }
                            continue
                        } else if mk.key == "spreads" {
                            // Spread: one pick per team with point as line
                            for out in m.outcomes {
                                let team = out.name
                                let line = out.point
                                let dedupeKey = "\(ev.id)|\(mk.key)|\(team)|\(line ?? -9999)"
                                if seenCombos.contains(dedupeKey) { continue }
                                seenCombos.insert(dedupeKey)

                                let oPrice = out.price
                                let implied = impliedProb(oPrice)
                                let conf = max(0.35, min(0.9,
                                                         0.5 + (abs(oPrice) >= 140 ? 0.18 : 0.0) + (implied > 0.55 ? 0.07 : -0.03)))
                                let tier = tier(for: conf)
                                var badges = [book.title]

                                picks.append(Pick(
                                    sport: sport,
                                    eventId: ev.id,
                                    eventLabel: label,
                                    commence: commence,
                                    player: team,                  // team name
                                    marketKey: mk,
                                    line: line,                    // spread number
                                    overOdds: oPrice,
                                    underOdds: nil,
                                    bookmaker: book.title,
                                    confidence: conf,
                                    tier: tier,
                                    badges: badges
                                ))
                            }
                            continue
                        }
                        // --- End team market special-cases ---

                        // group outcomes by (player, point)
                        var byPlayerPoint: [String: [APIOutcome]] = [:]
                        for out in m.outcomes {
                            let player = out.playerName
                            let pKey = "\(player)|\(out.point ?? -9999)"
                            byPlayerPoint[pKey, default: []].append(out)
                        }

                        for (combo, outs) in byPlayerPoint {
                            let parts = combo.split(separator: "|")
                            guard parts.count >= 2 else { continue }
                            // let player = String(parts[0])
                            let player = String(parts[0]).isEmpty && mk.key == "totals" ? "Game Total" : String(parts[0])
                            let line = Double(parts[1])

                            let over = outs.first { $0.isOver == true }
                            let under = outs.first { $0.isOver == false }
                            let any = outs.first

                            let overPrice = over?.price ?? any?.price
                            let underPrice = under?.price

                            guard let oPrice = overPrice else { continue }

                            let implied = impliedProb(oPrice)
                            let conf = max(0.35, min(0.9,
                                                     0.5 + (abs(oPrice) >= 140 ? 0.18 : 0.0) + (implied > 0.55 ? 0.07 : -0.03)))
                            let tier = tier(for: conf)
                            var badges: [String] = []
                            if line != nil && (line!.truncatingRemainder(dividingBy: 1) != 0) { badges.append("Alt Line") }
                            badges.append(book.title)

                            let dedupeKey = "\(ev.id)|\(mk.key)|\(player)|\(line ?? -9999)"
                            if seenCombos.contains(dedupeKey) { continue }
                            seenCombos.insert(dedupeKey)

                            picks.append(Pick(
                                sport: sport,
                                eventId: ev.id,
                                eventLabel: label,
                                commence: commence,
                                player: player,
                                marketKey: mk,
                                line: line,
                                overOdds: oPrice,
                                underOdds: underPrice,
                                bookmaker: book.title,
                                confidence: conf,
                                tier: tier,
                                badges: badges
                            ))
                        }
                    }
                }
            }

            // Optional market parity smoothing (gentle)
            if marketParityMode {
                let grouped = Dictionary(grouping: picks) { $0.marketKey.key + "|" + ($0.line?.description ?? "-") }
                picks = picks.map { p in
                    var p = p
                    if let g = grouped[p.marketKey.key + "|" + (p.line?.description ?? "-")], g.count > 6 {
                        let newConf = max(0.3, p.confidence - 0.03)
                        p = Pick(sport: p.sport, eventId: p.eventId, eventLabel: p.eventLabel, commence: p.commence, player: p.player, marketKey: p.marketKey, line: p.line, overOdds: p.overOdds, underOdds: p.underOdds, bookmaker: p.bookmaker, confidence: newConf, tier: tier(for: newConf), badges: p.badges)
                    }
                    return p
                }
            }

            return picks.sorted { (a, b) in
                if a.commence != b.commence { return a.commence < b.commence }
                return a.confidence > b.confidence
            }
        }

        private func tier(for conf: Double) -> Tier {
            switch conf {
            case ..<0.50: return .Lotto
            case 0.50..<0.62: return .Risky
            case 0.62..<0.70: return .Medium
            default: return .Safe
            }
        }
// MARK: - Math Helpers
        
        func impliedProb(_ americanOdds: Double) -> Double {
            if americanOdds > 0 { return 100 / (americanOdds + 100) }
            return (-americanOdds) / ((-americanOdds) + 100)
        }
        
        func americanToDecimal(_ american: Double) -> Double {
            if american > 0 { return 1.0 + american/100.0 }
            return 1.0 + 100.0/(-american)
        }
        
        /// Correlation adjusted parlay probability (rough)
        func parlayHitProbability(for legs: [Pick]) -> Double {
            guard !legs.isEmpty else { return 0 }
            let base = legs.map { impliedProb($0.overOdds) }.reduce(1.0, *)
            let groups = Dictionary(grouping: legs) { $0.eventId }
            let collisions = groups.values.map{ $0.count }.filter{ $0 > 1 }.reduce(0,+)
            let penalty = max(0.70, 1.0 - 0.06 * Double(collisions))
            let parity = marketParityMode ? 0.96 : 1.0
            return max(0.0, min(1.0, base * penalty * parity))
        }
        
        func parlayExpectedValue(stake: Double, legs: [Pick]) -> (hitProb: Double, payout: Double, ev: Double, decimalOdds: Double) {
            guard !legs.isEmpty else { return (0, 0, 0, 0) }
            let decimal = legs.map { americanToDecimal($0.overOdds) }.reduce(1.0, *)
            let hit = parlayHitProbability(for: legs)
            let payout = stake * decimal
            // EV as *net* expectation:
            // winProfit = stake * (decimal - 1)
            // EV = hit * winProfit - (1 - hit) * stake
            let winProfit = stake * max(0, decimal - 1.0)
            let ev = hit * winProfit - (1.0 - hit) * stake
            return (hit, payout, ev, decimal)
        }
        
        func suggestedUnitSize() -> Double {
            let base = bankroll * 0.01
            switch riskMode {
            case .Conservative: return base * 0.5
            case .Moderate: return base
            case .Aggressive: return base * 1.5
            }
        }
        
        // Hedge persistence helpers
        var hedgePositions: [String] {
            get { (try? JSONDecoder().decode([String].self, from: Data(hedgeJSON.utf8))) ?? [] }
            set {
                if let data = try? JSONEncoder().encode(newValue) { hedgeJSON = String(decoding: data, as: UTF8.self) }
                objectWillChange.send()
            }
        }

        // MARK: - Correlation helpers (heuristic + learned patterns)
        // Optional signals fetched from backend or derived locally. Keyed by eventId.
        struct CorrelationSignal: Decodable, Hashable {
            enum Kind: String, Decodable { case matchup_trend, head_to_head, team_style, pace_injury, news }
            let id: String
            let kind: Kind
            let eventId: String
            let players: [String]        // names involved (1 or 2)
            let teams: [String]          // home/away identifiers if any
            let markets: [String]        // affected market keys (e.g., ["player_points"]) or [] for all
            let boost: Double            // 0...0.35 typical
            let reason: String           // human string e.g. "Lillard vs Westbrook +8 pts avg last 6"
        }

        @Published var signalsByEventId: [String: [CorrelationSignal]] = [:]

        /// Try to fetch correlation signals from backend (non-fatal if missing)
        func fetchSignalsIfAvailable(for eventIds: [String]) async {
            await withTaskGroup(of: (String, [CorrelationSignal]).self) { group in
                for eid in Set(eventIds) {
                    group.addTask {
                        guard let url = URL(string: "\(BACKEND_BASE)/api/signals?event_id=\(eid)") else { return (eid, []) }
                        var req = URLRequest(url: url)
                        req.setValue("application/json", forHTTPHeaderField: "Accept")
                        do {
                            let (data, resp) = try await URLSession.shared.data(for: req)
                            guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else { return (eid, []) }
                            let arr = try JSONDecoder().decode([CorrelationSignal].self, from: data)
                            return (eid, arr)
                        } catch { return (eid, []) }
                    }
                }
                var dict: [String:[CorrelationSignal]] = [:]
                for await (eid, arr) in group { if !arr.isEmpty { dict[eid] = arr } }
                if !dict.isEmpty { await MainActor.run { self.signalsByEventId.merge(dict) { $0 + $1 } } }
            }
        }

        /// Build lightweight derived signals locally from current picks when backend has none.
        private func buildDerivedSignals() {
            var derived: [String:[CorrelationSignal]] = [:]
            // Rule 1: same-game, high-confidence shooter + facilitator ⇒ mild boost
            let groups = Dictionary(grouping: topPicks) { $0.eventId }
            for (eid, legs) in groups {
                let high = legs.filter { $0.confidence >= 0.68 }
                // scorer ↔ assists
                for s in high.filter({ $0.marketKey.key == "player_points" }) {
                    for a in high.filter({ $0.marketKey.key == "player_assists" && $0.player != s.player }) {
                        let sig = CorrelationSignal(id: UUID().uuidString,
                                                     kind: .team_style,
                                                     eventId: eid,
                                                     players: [s.player, a.player],
                                                     teams: [],
                                                     markets: ["player_points","player_assists"],
                                                     boost: 0.06,
                                                     reason: "High-usage scorer + facilitator in same game")
                        derived[eid, default: []].append(sig)
                    }
                }
                // Rule 2: QB + WR/TE yardage hints from markets present (NFL/CFB)
                let qbs = high.filter { $0.marketKey.key == "player_pass_yards" }
                let recs = high.filter { ["player_receiving_yards","player_receptions"].contains($0.marketKey.key) }
                for q in qbs { for r in recs where q.eventId == r.eventId { derived[eid, default: []].append(
                    CorrelationSignal(id: UUID().uuidString, kind: .matchup_trend, eventId: eid, players: [q.player, r.player], teams: [], markets: ["player_pass_yards","player_receiving_yards","player_receptions"], boost: 0.08, reason: "Passing volume supports receiver production")) } }
                // Rule 3: Pace/injury proxy — if many props in one game show high confidence, nudge intra-game correlations a bit
                if high.count >= 10 {
                    derived[eid, default: []].append(CorrelationSignal(id: UUID().uuidString, kind: .pace_injury, eventId: eid, players: [], teams: [], markets: [], boost: 0.04, reason: "Game-wide tailwind from pace/injuries"))
                }

                // Rule 4: High Total tailwind → points/assists/3PT lift (NBA-style)
                if let tot = legs.first(where: { $0.marketKey.key == "totals" }), let line = tot.line, line >= 228 {
                    let sig = CorrelationSignal(
                        id: UUID().uuidString,
                        kind: .pace_injury,
                        eventId: eid,
                        players: [],
                        teams: [],
                        markets: ["player_points","player_assists","player_threes"],
                        boost: 0.04,
                        reason: "High total (\(Int(line))) → pace/efficiency tailwind"
                    )
                    derived[eid, default: []].append(sig)
                }

                // Rule 5: Favorite ML → star usage bump (points)
                // Detect strong favorite from h2h price ≤ -150, then boost top 2 players by confidence for points.
                let moneylines = legs.filter { $0.marketKey.key == "h2h" }
                if let fav = moneylines.filter({ $0.overOdds <= -150 }).sorted(by: { abs($0.overOdds) > abs($1.overOdds) }).first {
                    // choose top 2 same-team player props by confidence (proxy: same event, exclude team picks)
                    let sameGamePlayers = legs.filter { $0.marketKey.key == "player_points" && $0.player != fav.player }
                                              .sorted(by: { $0.confidence > $1.confidence })
                                              .prefix(2)
                    for p in sameGamePlayers {
                        let sig = CorrelationSignal(
                            id: UUID().uuidString,
                            kind: .matchup_trend,
                            eventId: eid,
                            players: [p.player],
                            teams: [],
                            markets: ["player_points"],
                            boost: 0.03,
                            reason: "Favorite (\(fav.player)) → star scoring usage bump"
                        )
                        derived[eid, default: []].append(sig)
                    }
                }

                // Rule 6: Recent form proxy → same-player multi-market high confidence
                // If a player shows ≥2 markets with confidence ≥ 0.72 in this event, add a small boost on those markets.
                let byPlayer = Dictionary(grouping: legs) { $0.player }
                for (player, arr) in byPlayer {
                    let strong = arr.filter { $0.confidence >= 0.72 && $0.marketKey.key.hasPrefix("player_") }
                    if strong.count >= 2 {
                        let mk = Array(Set(strong.map { $0.marketKey.key }))
                        let sig = CorrelationSignal(
                            id: UUID().uuidString,
                            kind: .news,
                            eventId: eid,
                            players: [player],
                            teams: [],
                            markets: mk,
                            boost: 0.04,
                            reason: "Recent form / role stability across markets"
                        )
                        derived[eid, default: []].append(sig)
                    }
                }
            }
            signalsByEventId = derived
        }

        /// Aggregate applicable signals for a given pair of legs.
        func learnedBoost(_ a: Pick, _ b: Pick) -> (Double, String?) {
            guard a.eventId == b.eventId else { return (0, nil) }
            let arr = signalsByEventId[a.eventId] ?? []
            var best: (Double,String)? = nil
            for s in arr {
                // player targeting: s.players empty ⇒ global; otherwise must include at least one of each if two listed
                let playersOK: Bool = {
                    if s.players.isEmpty { return true }
                    if s.players.count == 1 { return s.players.contains(a.player) || s.players.contains(b.player) }
                    return s.players.contains(a.player) && s.players.contains(b.player)
                }()
                guard playersOK else { continue }
                // market filter: empty ⇒ any, else either leg in set
                let marketsOK = s.markets.isEmpty || s.markets.contains(a.marketKey.key) || s.markets.contains(b.marketKey.key)
                guard marketsOK else { continue }
                if best == nil || s.boost > (best!.0) { best = (s.boost, s.reason) }
            }
            return best ?? (0, nil)
        }

        /// Rough correlation score between two legs [0,1].
        func correlationScore(_ a: Pick, _ b: Pick) -> Double {
            var score = 0.0
            if a.eventId == b.eventId { score += 0.22 }                 // same game synergy
            if a.player == b.player { score += 0.32 }                   // same player, different stat
            if a.marketKey.key == b.marketKey.key { score += 0.05 }     // same market light boost
            if a.bookmaker == b.bookmaker { score += 0.03 }             // same book pricing
            if let la = a.line, let lb = b.line, abs(la - lb) <= 1.0 { score += 0.02 }
            // domain rules
            let (domainBonus, _) = marketPairBoost(a, b)
            score += domainBonus
            // learned signals (matchup/history/news)
            let (learned, _) = learnedBoost(a, b)
            score += learned
            return min(1.0, max(0.0, score))
        }

        /// UI helper: percent correlation 0-100 for a pair
        func correlationPercent(_ a: Pick, _ b: Pick) -> Int {
            return Int(correlationScore(a, b) * 100)
        }

        /// UI helper: best available reason for correlation (domain rule or learned signal)
        func correlationReason(_ a: Pick, _ b: Pick) -> String? {
            let (_, r1) = marketPairBoost(a, b)
            let (_, r2) = learnedBoost(a, b)
            return r1 ?? r2
        }

        /// Returns (bonus 0...1, reason string) for domain pairings.
        func marketPairBoost(_ a: Pick, _ b: Pick) -> (Double, String?) {
            guard a.eventId == b.eventId else { return (0.0, nil) }
            if let list = CORRELATION_RULES[a.marketKey.key], let m = list.first(where: { $0.other == b.marketKey.key }) { return (m.bonus, m.reason) }
            if let list = CORRELATION_RULES[b.marketKey.key], let m = list.first(where: { $0.other == a.marketKey.key }) { return (m.bonus, m.reason) }
            return (0.0, nil)
        }

        /// Suggestions for a leg based on same-game candidates not already in the slip.
        func suggestionsFor(leg: Pick, limit: Int = 5) -> [(Pick, Double, String?)] {
            let cands = topPicks.filter { $0.eventId == leg.eventId && $0.id != leg.id && !builderLegs.contains($0) }
            let scored: [(Pick, Double, String?)] = cands.map { cand in
                let base = correlationScore(leg, cand)
                let (_, reason1) = marketPairBoost(leg, cand)
                let (_, reason2) = learnedBoost(leg, cand)
                let reason = reason1 ?? reason2
                return (cand, base, reason)
            }
            return Array(scored.sorted { $0.1 > $1.1 }.prefix(limit))
        }
    }

        // MARK: - Root
    struct ContentView: View {
        @StateObject private var vm = AppVM()
        
        var body: some View {
            TabView {
                PicksHome(vm: vm)
                    .tabItem { Label("Picks", systemImage: "sparkles") }
                PropsBrowserView(vm: vm)
                    .tabItem { Label("Props", systemImage: "list.bullet.rectangle") }
                AIBetBuilder(vm: vm)
                    .tabItem { Label("AI Builder", systemImage: "wand.and.stars") }
                PlannerView(vm: vm)
                    .tabItem { Label("Planner", systemImage: "chart.pie.fill") }
                HeatMapView(vm: vm)
                    .tabItem { Label("Heat Map", systemImage: "flame") }
                // MARK: - Settings (placeholder to restore build)
                SettingsView(vm: vm)
                    .tabItem { Label("Settings", systemImage: "gearshape") }
            }
        }
    }
    
// MARK: - Picks Home

struct PicksHome: View {
    @ObservedObject var vm: AppVM
    @State private var selectedTier: Tier? = nil
    @State private var showingHedge = false
    @State private var selectedEventId: String? = nil

    var filteredPicks: [Pick] {
        vm.topPicks.filter {
            (selectedTier == nil || $0.tier == selectedTier!) &&
            (selectedEventId == nil || $0.eventId == selectedEventId!)
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                header
                creditsBanner
                riskRow
                actionRow
                content
            }
            .padding(.horizontal).padding(.top, 8)
            .navigationTitle("SmartPicks Pro")
            .navigationBarTitleDisplayMode(.inline)          // keeps title from covering tabs
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Hedge Monitor") { showingHedge = true }
                }
            }
            .sheet(isPresented: $showingHedge) { HedgeMonitorView(vm: vm) }
            .task { await vm.refresh() }
            .scrollDismissesKeyboard(.interactively)
            .modifier(BetSlipOverlay(vm: vm))
        }
    }

    // Header with sport/date + game/tier filters
    private var header: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Sport", selection: $vm.selectedSport) {
                    ForEach(Sport.allCases) { s in Text(s.rawValue).tag(s) }
                }
                .pickerStyle(.segmented)
                .onChange(of: vm.selectedSport) { _ in Task { await vm.refresh() } }

                HStack {
                    DatePicker("Date", selection: $vm.date, displayedComponents: .date)
                        .datePickerStyle(.compact)
                        .onChange(of: vm.date) { _ in Task { await vm.refresh() } }
                    Toggle("All Upcoming", isOn: $vm.showAllUpcoming)
                        .toggleStyle(.switch)
                }

                // Game selector
                let games = Array(Dictionary(grouping: vm.topPicks, by: { $0.eventId }))
                    .map { (id: $0.key, label: $0.value.first?.eventLabel ?? "Unknown") }
                    .sorted { $0.label < $1.label }

                Picker("Game", selection: $selectedEventId) {
                    Text("All Games").tag(String?.none)
                    ForEach(games, id: \.id) { g in
                        Text(g.label).tag(String?.some(g.id))
                    }
                }
                .pickerStyle(.menu)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        tierChip(nil, label: "All")
                        ForEach(Tier.allCases) { t in tierChip(t, label: t.rawValue) }
                    }
                }
                HStack {
                    NavigationLink(destination: PropsBrowserView(vm: vm)) {
                        Label("Browse All Props", systemImage: "list.bullet")
                            .font(.footnote.weight(.semibold))
                    }
                    Spacer()
                }
                // Status: last updated time from the odds client
                if let t = OddsAPIClient.shared.lastUpdated {
                    HStack(spacing: 6) {
                        Image(systemName: "clock")
                        Text("Updated \(t, style: .relative)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }
            }
        }
        .padding(.top, 4)
        .padding(.bottom, 4)
        .padding(.bottom, 6)
    }

    private func tierChip(_ tier: Tier?, label: String) -> some View {
        let isOn = selectedTier?.rawValue == tier?.rawValue || (tier == nil && selectedTier == nil)
        return Button(action: { selectedTier = tier }) {
            Text(label)
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 12).padding(.vertical, 6)
                .background(Capsule().fill(isOn ? Color.accentColor.opacity(0.18)
                                                : Color.secondary.opacity(0.12)))
        }.buttonStyle(.plain)
    }

    private var creditsBanner: some View {
        Group {
            if let remaining = OddsAPIClient.shared.requestsRemaining, remaining <= 5 {
                HStack(spacing: 8) {
                    Image(systemName: "bolt.horizontal.circle")
                    Text("Low API credits: \(remaining) left").font(.footnote)
                    Spacer()
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.orange.opacity(0.12)))
            }
        }
    }

    private var riskRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "shield.lefthalf.filled").foregroundStyle(.blue)
            Picker("Risk", selection: Binding(get: { vm.riskMode }, set: { vm.riskMode = $0 })) {
                ForEach(RiskMode.allCases) { r in Text(r.rawValue).tag(r) }
            }.pickerStyle(.segmented)
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("Bankroll: $\(Int(vm.bankroll))").font(.footnote)
                Text("Unit: $\(Int(vm.suggestedUnitSize()))").font(.caption2).foregroundStyle(.secondary)
                Text("Risk affects unit size & AI picks").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }


    // Expandable / copyable error message
    private var actionRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button { Task { await vm.refresh() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                    .buttonStyle(.bordered)
                if vm.isLoading { ProgressView().progressViewStyle(.circular) }
                Spacer()
            }
            if let msg = vm.errorMessage {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("Odds fetch failed").font(.caption.bold())
                        Spacer()
                        Button("Copy") { UIPasteboard.general.string = msg }
                    }
                    ScrollView {
                        Text(msg)
                            .font(.system(size: 12, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 160)
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.red.opacity(0.12)))
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color(.secondarySystemBackground)))
    }

    @ViewBuilder private var content: some View {
        if vm.topPicks.isEmpty {
            VStack(spacing: 12) {
                Spacer(minLength: 20)
                Text(vm.isLoading ? "Loading picks..." : "No picks found for selection.")
                    .foregroundStyle(.secondary)
                Spacer(minLength: 12)
            }
        } else {
            Text("Top Picks")
                .font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.bottom, 4)

            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(filteredPicks) { pick in
                        PickRow(
                            pick: pick,
                            addAction: { if !vm.builderLegs.contains(pick) { vm.builderLegs.append(pick) } },
                            vm: vm
                        )
                    }
                }
            }
            .scrollIndicators(.hidden)
            .safeAreaPadding(.bottom, 80)
        }
    }

}
   // MARK: - Components
    
    struct LossCapBanner: View {
        let amount: Double
        var body: some View {
            HStack(alignment: .top) {
                Image(systemName: "exclamationmark.triangle.fill")
                VStack(alignment: .leading, spacing: 4) {
                    Text("Responsible Gaming").font(.subheadline.weight(.semibold))
                    Text("Daily loss cap set to $\(Int(amount)). We'll warn you when your risk exceeds this cap.").font(.caption)
                }
                Spacer()
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 12).fill(Color.yellow.opacity(0.18)))
        }
    }
    
    struct BadgeView: View {
        let text: String
        var body: some View { Text(text).font(.caption2.weight(.semibold)).padding(.horizontal, 8).padding(.vertical, 4).background(Capsule().fill(Color.secondary.opacity(0.12))) }
    }
    
struct PickRow: View {
    let pick: Pick
    var addAction: () -> Void
    /// Pass vm so we can show live correlation vs current slip
    var vm: AppVM? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(pick.player).font(.headline).lineLimit(1)
                Spacer()
                TierTag(tier: pick.tier)
            }

            HStack(spacing: 8) {
                pill(text: pick.marketKey.label)
                if let ln = pick.line { pill(text: String(format: "%.1f", ln)) }
                if pick.marketKey.key == "totals" || pick.underOdds != nil {
                    // Totals or true O/U props: show both sides when available
                    let text = pick.underOdds != nil
                        ? "O \(oddsString(pick.overOdds)) / U \(oddsString(pick.underOdds!))"
                        : "O \(oddsString(pick.overOdds))"
                    pill(text: text)
                } else {
                    // Moneyline / Spread or single-sided props: show single odds without O/U prefix
                    pill(text: oddsString(pick.overOdds))
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                BadgeView(text: pick.eventLabel)
                BadgeView(text: pick.bookmaker)
                Spacer()
                ConfidenceBar(value: pick.confidence)
            }

            // 🔗 Correlation hint vs best current leg from same game
            if let vm = vm {
                let sameGame = vm.builderLegs.filter { $0.eventId == pick.eventId }
                if let best = sameGame
                    .map({ ($0, vm.correlationScore(pick, $0), vm.correlationReason(pick, $0)) })
                    .sorted(by: { $0.1 > $1.1 })
                    .first, best.1 > 0.10 {

                    HStack(spacing: 6) {
                        Image(systemName: "link")
                        Text("Best match: \(best.0.player) — \(best.0.marketKey.label)")
                            .lineLimit(1)
                            .font(.caption)
                        Spacer()
                        Text("\(Int(best.1 * 100))%")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Capsule().fill(Color.secondary.opacity(0.12)))
                    }
                    .foregroundStyle(.secondary)
                    if let reason = best.2 {
                        Text(reason).font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }

            HStack {
                Button(action: addAction) { Label("Add", systemImage: "plus") }
                    .buttonStyle(.borderedProminent)
                Spacer()
                Text(pick.commence, style: .time)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 16).fill(.ultraThinMaterial))
        .contentShape(Rectangle())
        .onTapGesture { addAction() }
    }

    private func pill(text: String) -> some View {
        Text(text)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(Capsule().fill(Color.secondary.opacity(0.12)))
    }
    private func oddsString(_ odds: Double) -> String { odds > 0 ? "+\(Int(odds))" : "\(Int(odds))" }
}
    
    struct TierTag: View {
        let tier: Tier
        var color: Color { switch tier { case .Safe: .green; case .Medium: .blue; case .Risky: .orange; case .Lotto: .purple } }
        var body: some View { Text(tier.rawValue).font(.caption2.weight(.bold)).padding(.horizontal, 8).padding(.vertical, 4).foregroundStyle(.white).background(Capsule().fill(color)) }
    }
    
    struct ConfidenceBar: View {
        let value: Double // 0...1
        var body: some View {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.15))
                    RoundedRectangle(cornerRadius: 6).fill(Color.accentColor).frame(width: max(0, min(geo.size.width * value, geo.size.width)))
                }
            }
            .frame(width: 100, height: 8)
            .overlay(alignment: .trailing) { Text("\(Int(value * 100))%").font(.caption2).foregroundStyle(.secondary) }
        }
    }
    
    struct ParlayComposer: View {
        @ObservedObject var vm: AppVM
        @State private var stakeText: String = "10"
        
        private var stake: Double {
            let v = Double(stakeText) ?? 0
            return max(0, v)
        }
        
        var body: some View {
            VStack(alignment: .leading, spacing: 10) {
                
                // Stake typed input + Clear
                HStack {
                    TextField("Stake ($)", text: $stakeText)
                        .keyboardType(.decimalPad)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 160)
                    
                    Spacer()
                    Button("Clear") { vm.builderLegs.removeAll() }
                        .buttonStyle(.bordered)
                }
                
                if vm.builderLegs.isEmpty {
                    Text("No legs yet. Tap \"Add to Parlay\" on a pick to include it.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(vm.builderLegs) { leg in
                        HStack {
                            Text("\(leg.player) — \(leg.marketKey.label) \(leg.line?.formatted() ?? "")")
                                .lineLimit(1)
                            Spacer()
                            Text(leg.overOdds > 0 ? "+\(Int(leg.overOdds))" : "\(Int(leg.overOdds))")
                        }
                        .font(.caption)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
                    }
                    
                    let res = vm.parlayExpectedValue(stake: stake, legs: vm.builderLegs)
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Decimal Odds: \(String(format: "%.2f", res.decimalOdds))")
                        Text("Hit Probability: \(Int(res.hitProb * 100))%")
                        Text(String(format: "Projected Payout: $%.2f", res.payout))
                        Text(String(format: "Expected Value: $%.2f", res.ev))
                            .foregroundStyle(res.ev >= 0 ? .green : .red)
                    }
                    .font(.footnote)
                    
                    // Correlation info
                    let sameGameGroups = Dictionary(grouping: vm.builderLegs, by: { $0.eventId })
                    let multiFromSame = sameGameGroups.values.filter { $0.count > 1 }.count
                    if multiFromSame > 0 {
                        Text("Correlation: \(multiFromSame) game(s) have multiple legs. A penalty is applied to hit %.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    // Suggested correlated add-ons
                    let currentEventIds = Set(vm.builderLegs.map { $0.eventId })
                    let suggested = vm.topPicks
                        .filter { currentEventIds.contains($0.eventId) && !vm.builderLegs.contains($0) }
                        .sorted { $0.confidence > $1.confidence }
                        .prefix(6)
                    
                    if !suggested.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Correlated adds").font(.subheadline.weight(.semibold))
                            ForEach(Array(suggested), id: \.id) { sug in
                                HStack {
                                    Text("\(sug.player) — \(sug.marketKey.label) \(sug.line?.formatted() ?? "")")
                                        .font(.caption)
                                        .lineLimit(1)
                                    Spacer()
                                    Button("+") {
                                        if !vm.builderLegs.contains(sug) { vm.builderLegs.append(sug) }
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }
                        }
                    }
                    
                    if vm.dailyLossCapEnabled && (stake > vm.dailyLossCapAmount) {
                        Text("Warning: Stake exceeds daily loss cap of $\(Int(vm.dailyLossCapAmount)).")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }
            // universal keyboard Done for this view
            .keyboardToolbarDone {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
    }

// MARK: - Props Browser (MVP)
struct PropsBrowserView: View {
    @ObservedObject var vm: AppVM
    @State private var selectedEventId: String? = nil
    @State private var selectedMarket: String? = nil
    @State private var searchText: String = ""

    private var games: [(id: String, label: String)] {
        Array(Dictionary(grouping: vm.topPicks, by: { $0.eventId }))
            .map { (id: $0.key, label: $0.value.first?.eventLabel ?? "Unknown") }
            .sorted { $0.label < $1.label }
    }

    // Local fallback list to avoid dependency on global constants
    private let ALL_PROP_MARKETS_LOCAL: [String] = [
        // NBA-like
        "player_points","player_rebounds","player_assists","player_threes","player_steals","player_blocks","player_turnovers",
        "player_points_rebounds_assists","player_points_rebounds","player_points_assists","player_rebounds_assists",
        // NFL/CFB
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds","player_receiving_yards","player_receptions",
        "player_receiving_tds","player_longest_reception","player_longest_rush",
        // MLB
        "player_hits","player_runs","player_rbis","player_home_runs","player_total_bases","player_walks",
        "player_strikeouts","pitcher_outs","pitcher_hits_allowed",
        // NHL
        "player_shots_on_goal","player_goals","player_assists","player_points","goalie_saves",
        // Team markets
        "h2h","spreads","totals"
    ]

    private var marketKeys: [String] {
        let discovered = Set(vm.topPicks.map { $0.marketKey.key })
        let base = discovered.union(Set(ALL_PROP_MARKETS_LOCAL))
        return Array(base).sorted()
    }

    private var filtered: [Pick] {
        var arr = vm.topPicks
        if let eid = selectedEventId { arr = arr.filter { $0.eventId == eid } }
        if let mk = selectedMarket { arr = arr.filter { $0.marketKey.key == mk } }
        if !searchText.isEmpty {
            let q = searchText.lowercased()
            arr = arr.filter { $0.player.lowercased().contains(q) }
        }
        return arr.sorted { a, b in
            if a.commence != b.commence { return a.commence < b.commence }
            if a.marketKey.key != b.marketKey.key { return a.marketKey.key < b.marketKey.key }
            return a.player < b.player
        }
    }

    // Local label override for team markets to avoid touching global MarketKey
    private func friendlyMarketLabel(_ key: String) -> String {
        switch key {
        case "h2h":    return "Moneyline"
        case "spreads": return "Spread"
        case "totals":  return "Total"
        default:
            return MarketKey(key: key).label
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 10) {
                HStack(spacing: 8) {
                    Picker("Game", selection: $selectedEventId) {
                        Text("All Games").tag(String?.none)
                        ForEach(games, id: \.id) { g in
                            Text(g.label).tag(String?.some(g.id))
                        }
                    }.pickerStyle(.menu)

                    Picker("Category", selection: $selectedMarket) {
                        Text("All Props").tag(String?.none)
                        ForEach(marketKeys, id: \.self) { k in
                            Text(friendlyMarketLabel(k)).tag(String?.some(k))
                        }
                    }.pickerStyle(.menu)
                }

                TextField("Search player", text: $searchText)
                    .textFieldStyle(.roundedBorder)

                if filtered.isEmpty {
                    Spacer(minLength: 20)
                    Text("No props match your filters.")
                        .foregroundStyle(.secondary)
                    Spacer()
                } else {
                    List {
                        ForEach(filtered) { p in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(p.player).font(.subheadline.weight(.semibold))
                                    Spacer()
                                    Text(p.eventLabel).font(.caption).foregroundStyle(.secondary)
                                }
                                HStack(spacing: 8) {
                                    pill(p.marketKey.label)
                                    if let ln = p.line { pill(String(format: "%.1f", ln)) }
                                    if p.marketKey.key == "totals" || p.underOdds != nil {
                                        let text = p.underOdds != nil
                                            ? "O \(oddsString(p.overOdds)) / U \(oddsString(p.underOdds!))"
                                            : "O \(oddsString(p.overOdds))"
                                        pill(text)
                                    } else {
                                        pill(oddsString(p.overOdds))
                                    }
                                    Spacer()
                                    Button("Add") {
                                        if !vm.builderLegs.contains(p) { vm.builderLegs.append(p) }
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }
                            .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .padding()
            .navigationTitle("Props Browser")
            .modifier(BetSlipOverlay(vm: vm))
        }
    }
}

// MARK: - Bet Slip (FanDuel-style)

struct BetSlipBar: View {
    @ObservedObject var vm: AppVM
    var expand: () -> Void

    var body: some View {
        Button(action: expand) {
            HStack(spacing: 12) {
                Image(systemName: "ticket.fill")
                Text("Bet Slip").font(.subheadline.weight(.semibold))
                Text("• \(vm.builderLegs.count) leg\(vm.builderLegs.count == 1 ? "" : "s")")
                    .foregroundStyle(.secondary)
                Spacer()
                // quick metrics on 1 suggested unit
                let stake = max(1.0, vm.suggestedUnitSize())
                let res = vm.parlayExpectedValue(stake: stake, legs: vm.builderLegs)
                Text("\(Int(res.hitProb * 100))%")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Capsule().fill(Color.secondary.opacity(0.15)))
                Image(systemName: "chevron.up")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
        .buttonStyle(.plain)
    }
}

struct BetSlipPanel: View {
    @ObservedObject var vm: AppVM
    @Environment(\.dismiss) private var dismiss
    @State private var stakeText: String = "10"
    @State private var didSetDefaultStake = false
    private var stake: Double { max(0, Double(stakeText) ?? 0) }

    var body: some View {
        GroupBox {
            VStack(spacing: 12) {
                HStack {
                    Text("Bet Slip").font(.headline)
                    Spacer()
                    Button("Done") { dismiss() }
                        .buttonStyle(.bordered)
                    Button("Clear") { vm.builderLegs.removeAll() }
                        .buttonStyle(.bordered)
                }

                HStack {
                    TextField("Stake ($)", text: $stakeText)
                        .keyboardType(.decimalPad)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 160)
                    Spacer()
                }
                HStack(spacing: 8) {
                    Button("Unit") { stakeText = String(Int(max(1, vm.suggestedUnitSize()))) }
                        .buttonStyle(.bordered)
                    Button("2x") { stakeText = String(Int(max(1, vm.suggestedUnitSize() * 2))) }
                        .buttonStyle(.bordered)
                    Button("5x") { stakeText = String(Int(max(1, vm.suggestedUnitSize() * 5))) }
                        .buttonStyle(.bordered)
                }
                .font(.caption)

                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(vm.builderLegs) { leg in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(leg.player).font(.subheadline.weight(.semibold)).lineLimit(1)
                                    Spacer()
                                    Button(role: .destructive) {
                                        vm.builderLegs.removeAll { $0.id == leg.id }
                                    } label: { Image(systemName: "trash") }
                                }
                                HStack(spacing: 8) {
                                    pill(leg.marketKey.label)
                                    if let ln = leg.line { pill(String(format: "%.1f", ln)) }
                                    pill(leg.overOdds > 0 ? "+\(Int(leg.overOdds))" : "\(Int(leg.overOdds))")
                                    pill(leg.bookmaker)
                                    Spacer()
                                    ConfidenceBar(value: leg.confidence).frame(width: 80)
                                }
                                .font(.caption).foregroundStyle(.secondary)

                                // Correlations inside the slip (detailed pair callouts)
                                let sameGame = vm.builderLegs.filter { $0.eventId == leg.eventId && $0.id != leg.id }
                                if !sameGame.isEmpty {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack(spacing: 6) { Image(systemName: "link"); Text("Correlated legs") }
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(.secondary)
                                        ForEach(sameGame) { g in
                                            VStack(alignment: .leading, spacing: 2) {
                                                HStack(spacing: 6) {
                                                    Text("\(g.player) — \(g.marketKey.label)")
                                                        .font(.caption)
                                                        .lineLimit(1)
                                                    Spacer()
                                                    Text("\(vm.correlationPercent(leg, g))%")
                                                        .font(.caption2)
                                                        .foregroundStyle(.secondary)
                                                }
                                                if let reason = vm.correlationReason(leg, g) {
                                                    Text(reason)
                                                        .font(.caption2)
                                                        .foregroundStyle(.secondary)
                                                }
                                            }
                                            .padding(6)
                                            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.06)))
                                        }
                                    }
                                }

                                // Suggestions with correlation percentage
                                let sugg = vm.suggestionsFor(leg: leg, limit: 3)
                                if !sugg.isEmpty {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Suggested adds").font(.caption.weight(.semibold))
                                        ForEach(Array(sugg.enumerated()), id: \.element.0.id) { _, triplet in
                                            let pick = triplet.0
                                            let corr = triplet.1
                                            let reason = triplet.2
                                            VStack(alignment: .leading, spacing: 2) {
                                                HStack {
                                                    Text("\(pick.player) — \(pick.marketKey.label)")
                                                        .font(.caption)
                                                        .lineLimit(1)
                                                    Spacer()
                                                    Text("\(Int(corr * 100))%")
                                                        .font(.caption2)
                                                        .foregroundStyle(.secondary)
                                                    Button("+") {
                                                        if !vm.builderLegs.contains(pick) { vm.builderLegs.append(pick) }
                                                    }
                                                    .buttonStyle(.bordered)
                                                }
                                                if let reason = reason {
                                                    Text(reason)
                                                        .font(.caption2)
                                                        .foregroundStyle(.secondary)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            .padding(10)
                            .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.08)))
                        }
                    }
                    .padding(.vertical, 2)
                }

                let res = vm.parlayExpectedValue(stake: stake, legs: vm.builderLegs)
                VStack(alignment: .leading, spacing: 4) {
                    HStack { Text("Decimal Odds:"); Spacer(); Text(String(format: "%.2f", res.decimalOdds)) }
                    HStack { Text("Hit Probability:"); Spacer(); Text("\(Int(res.hitProb * 100))%") }
                    HStack { Text("Projected Payout:"); Spacer(); Text(String(format: "$%.2f", res.payout)) }
                    HStack {
                        Text("Expected Value:"); Spacer()
                        Text(String(format: "$%.2f", res.ev)).foregroundStyle(res.ev >= 0 ? .green : .red)
                    }
                }
                .font(.footnote)
                .padding()
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.08)))

                Button { /* hook up later */ } label: {
                    Text("Place Bet (demo)").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .keyboardToolbarDone()
        .hideKeyboardOnTap()
        .onAppear {
            if !didSetDefaultStake {
                stakeText = String(Int(max(1, vm.suggestedUnitSize())))
                didSetDefaultStake = true
            }
        }
    }
}

struct BetSlipOverlay: ViewModifier {
    @ObservedObject var vm: AppVM
    @State private var showSlip = false
    func body(content: Content) -> some View {
        content
            .safeAreaInset(edge: .bottom) {
                if !vm.builderLegs.isEmpty {
                    BetSlipBar(vm: vm) { showSlip = true }
                        .background(.ultraThinMaterial)
                        .shadow(radius: 3)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.easeInOut(duration: 0.25), value: vm.builderLegs.count)
            .sheet(isPresented: $showSlip) {
                BetSlipPanel(vm: vm)
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            }
    }
}

// tiny helper reused above
@ViewBuilder private func pill(_ text: String) -> some View {
    Text(text).padding(.horizontal, 8).padding(.vertical, 4)
        .background(Capsule().fill(Color.secondary.opacity(0.12)))
}
    // MARK: - AI Builder (heuristic using live picks)
    
    struct AIBetBuilder: View {
        @ObservedObject var vm: AppVM
        @State private var prompt: String = "Describe any constraints (optional)."
        @State private var legCount: Int = 3

        var body: some View {
            NavigationStack {
                VStack(alignment: .leading, spacing: 12) {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 100)
                        .padding(8)
                        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.secondary.opacity(0.3)))
                        .keyboardToolbarDone()

                    HStack(spacing: 12) {
                        Stepper("Legs: \(legCount)", value: $legCount, in: 2...10)
                        Spacer()
                        Button("Generate") { generate() }
                            .buttonStyle(.borderedProminent)
                        Button("Clear") { vm.builderLegs.removeAll() }
                    }

                    if vm.builderLegs.isEmpty {
                        Text("Choose leg count and tap Generate — we'll optimize for confidence, correlation, and diversity.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    ScrollView { LazyVStack(spacing: 10) { ForEach(vm.builderLegs) { PickRow(pick: $0, addAction: {}) } } }
                }
                .padding()
                .navigationTitle("AI Bet Builder")
                .scrollDismissesKeyboard(.interactively)
                .hideKeyboardOnTap()
                .modifier(BetSlipOverlay(vm: vm))
            }
        }

        private func generate() {
            let pool = vm.topPicks
            guard !pool.isEmpty else { return }

            // Weights based on risk mode
            let (wConf, wCorr): (Double, Double) = {
                switch vm.riskMode {
                case .Conservative: return (0.75, 0.25)
                case .Moderate: return (0.65, 0.35)
                case .Aggressive: return (0.55, 0.45)
                }
            }()

            // Pick the event cluster with most high-confidence legs first
            let groups = Dictionary(grouping: pool) { $0.eventId }
            func clusterScore(_ arr: [Pick]) -> Double { arr.sorted { $0.confidence > $1.confidence }.prefix(8).map { $0.confidence }.reduce(0,+) }
            let seed = (groups.max { clusterScore($0.value) < clusterScore($1.value) }?.value ?? pool).sorted { $0.confidence > $1.confidence }

            var chosen: [Pick] = []
            var candidates = seed

            while chosen.count < legCount && !candidates.isEmpty {
                // score each candidate vs the current set
                let scored: [(Pick, Double)] = candidates.map { cand in
                    let corr = chosen.isEmpty ? 0.0 : chosen.map { vm.correlationScore($0, cand) }.reduce(0, +) / Double(chosen.count)
                    let diversityPenalty: Double = chosen.contains(where: { $0.player == cand.player && $0.marketKey.key == cand.marketKey.key }) ? 0.1 : 0.0
                    let composite = wConf * cand.confidence + wCorr * corr - diversityPenalty
                    return (cand, composite)
                }
                guard let best = scored.max(by: { $0.1 < $1.1 }) else { break }
                chosen.append(best.0)
                candidates.removeAll { $0.id == best.0.id }
            }

            vm.builderLegs = Array(chosen.prefix(legCount))
        }
    }
    
// MARK: - Heat Map (derived from live picks)

struct HeatMapView: View {
    @ObservedObject var vm: AppVM
    @State private var selectedEventId: String? = nil

    // Choose a few key markets per sport for columns
    private var columns: [MarketKey] {
        switch vm.selectedSport {
        case .NBA: return ["player_points","player_assists","player_rebounds","player_threes"].map { MarketKey(key: $0) }
        case .NFL, .CFB: return ["player_pass_yards","player_rush_yards","player_receiving_yards","player_receptions"].map { MarketKey(key: $0) }
        case .MLB: return ["player_total_bases","player_hits","player_runs","player_rbis"].map { MarketKey(key: $0) }
        case .NHL: return ["player_points","player_assists","player_shots_on_goal"].map { MarketKey(key: $0) }
        }
    }

    // Build a map: player -> market -> best confidence
    private var grid: [(player: String, values: [String: Double])] {
        let source = selectedEventId == nil ? vm.topPicks : vm.topPicks.filter { $0.eventId == selectedEventId! }
        var dict: [String: [String: Double]] = [:]
        for p in source {
            var row = dict[p.player, default: [:]]
            row[p.marketKey.key] = max(row[p.marketKey.key] ?? 0.0, p.confidence)
            dict[p.player] = row
        }
        // Rank players by max across shown columns
        let ranked = dict.map { (k,v) in (k,v, columns.map{ v[$0.key] ?? 0 }.max() ?? 0) }
            .sorted { $0.2 > $1.2 }
        return ranked.map { ($0.0, $0.1) }
    }

    private func cellColor(_ v: Double) -> Color {
        // map 0 (cold) -> red, 0.5 neutral -> gray, 1 (hot) -> green
        let green = v
        let red = 1 - v
        return Color(red: red, green: green, blue: 0.2)
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 8) {
                // Game menu
                let games = Array(Dictionary(grouping: vm.topPicks, by: { $0.eventId }))
                    .map { (id: $0.key, label: $0.value.first?.eventLabel ?? "Unknown") }
                    .sorted { $0.label < $1.label }
                Picker("Game", selection: $selectedEventId) {
                    Text("All Games").tag(String?.none)
                    ForEach(games, id: \.id) { g in Text(g.label).tag(String?.some(g.id)) }
                }
                .pickerStyle(.menu)

                ScrollView(.horizontal) {
                    VStack(alignment: .leading, spacing: 6) {
                        // Header row
                        HStack(spacing: 8) {
                            Text("Player").frame(width: 140, alignment: .leading).font(.caption.bold())
                            ForEach(columns, id: \.key) { mk in Text(mk.label).frame(width: 90, alignment: .center).font(.caption.bold()) }
                        }
                        Divider()
                        // Rows
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 6) {
                                ForEach(grid.prefix(40), id: \.player) { row in
                                    HStack(spacing: 8) {
                                        let maxV = columns.map { row.values[$0.key] ?? 0 }.max() ?? 0
                                        Text(row.player + (maxV >= 0.65 ? " 🔥" : (maxV <= 0.45 ? " 🥶" : "")))
                                            .frame(width: 140, alignment: .leading)
                                            .font(.caption)
                                            .lineLimit(1)
                                        ForEach(columns, id: \.key) { mk in
                                            let v = row.values[mk.key] ?? 0
                                            RoundedRectangle(cornerRadius: 6)
                                                .fill(cellColor(v))
                                                .overlay(Text(v == 0 ? "—" : "\(Int(v*100))%").font(.caption2))
                                                .frame(width: 90, height: 22)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding(.vertical, 6)
                }
            }
            .padding()
            .navigationTitle("Heat Map")
            .modifier(BetSlipOverlay(vm: vm))
        }
    }
}
    
    // MARK: - Hedge Monitor (persisted)
    
struct HedgeMonitorView: View {
    @ObservedObject var vm: AppVM
    @State private var newEntry: String = ""
    
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    TextField("Add open position (free text)", text: $newEntry)
                        .textFieldStyle(.roundedBorder)
                    Button("Add") {
                        if !newEntry.isEmpty {
                            var arr = vm.hedgePositions
                            arr.append(newEntry)
                            vm.hedgePositions = arr
                            newEntry = ""
                        }
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                
                List {
                    Section("Open Positions") {
                        ForEach(vm.hedgePositions, id: \.self) { pos in
                            Text(pos)
                        }
                        .onDelete { idx in
                            var arr = vm.hedgePositions
                            arr.remove(atOffsets: idx)
                            vm.hedgePositions = arr
                        }
                    }
                }
            }
            .navigationTitle("Hedge Monitor")
        }
    }
}

// MARK: - Planner v2 (bankroll → primaries + safety, singles or parlays)
struct PlannerView: View {
    @ObservedObject var vm: AppVM
    @State private var bankrollInput: String = ""
    @State private var targetProfitInput: String = ""
    @State private var legsPrimary: Int = 3
    @State private var legsSecondary: Int = 5
    @State private var legsSafety: Int = 2
    @State private var plan: [BetPlan] = []

    struct BetPlan: Identifiable, Hashable {
        let id = UUID()
        let legs: [Pick]              // 1+ legs
        let stake: Double
        let category: Tier            // Safe / Medium / Risky / Lotto
        let rationale: String
        // snapshot metrics (computed once when created)
        let hitProb: Double
        let decimalOdds: Double
        let payout: Double
        let ev: Double
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Goal") {
                    TextField("Bankroll ($)", text: $bankrollInput)
                        .keyboardType(.decimalPad)
                    TextField("Target Profit ($)", text: $targetProfitInput)
                        .keyboardType(.decimalPad)
                    HStack {
                        Stepper("Primary legs: \(legsPrimary)", value: $legsPrimary, in: 1...8)
                        Stepper("Second legs: \(legsSecondary)", value: $legsSecondary, in: 1...8)
                        Stepper("Safety legs: \(legsSafety)", value: $legsSafety, in: 1...8)
                    }
                    Button("Create Plan") { createPlan() }
                        .buttonStyle(.borderedProminent)
                }

                Section("Suggested Plan") {
                    if plan.isEmpty {
                        Text("Enter bankroll/target then Create Plan. We'll build two primaries plus a safety (single or parlay) sized to cover losses and chase the profit based on your risk mode.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(plan) { leg in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(leg.category.rawValue)
                                        .font(.caption2.weight(.bold))
                                        .padding(.horizontal, 8).padding(.vertical, 3)
                                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                                    Spacer()
                                    Text(String(format: "Stake $%.0f", leg.stake))
                                }
                                // Legs list
                                ForEach(leg.legs) { p in
                                    HStack(spacing: 8) {
                                        Text("\(p.player) — \(p.marketKey.label) \(p.line?.formatted() ?? "")")
                                            .font(.caption)
                                            .lineLimit(1)
                                        Spacer()
                                        Text(p.overOdds > 0 ? "+\(Int(p.overOdds))" : "\(Int(p.overOdds))")
                                            .font(.caption2)
                                    }
                                }
                                // Metrics
                                HStack {
                                    Text("Hit: \(Int(leg.hitProb * 100))%")
                                    Spacer()
                                    Text("Dec: \(String(format: "%.2f", leg.decimalOdds))")
                                    Spacer()
                                    Text("Payout: \(String(format: "$%.2f", leg.payout))")
                                    Spacer()
                                    Text("EV: \(String(format: "$%.2f", leg.ev))")
                                        .foregroundStyle(leg.ev >= 0 ? .green : .red)
                                }.font(.caption)
                                if !leg.rationale.isEmpty { Text(leg.rationale).font(.caption2).foregroundStyle(.secondary) }
                                HStack {
                                    Button("Add to Bet Slip") {
                                        for p in leg.legs { if !vm.builderLegs.contains(p) { vm.builderLegs.append(p) } }
                                    }.buttonStyle(.bordered)
                                    Spacer()
                                }
                            }
                            .padding(10)
                            .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.06)))
                        }
                    }
                }
            }
            .navigationTitle("Bankroll Planner")
            .modifier(BetSlipOverlay(vm: vm))
        }
    }

    // MARK: - Planning logic
    private func createPlan() {
        guard let bankroll = Double(bankrollInput), bankroll > 0 else { plan = []; return }
        let target = max(0, Double(targetProfitInput) ?? 0)

        // Budget split: 40% + 30% + 30% (primary A / primary B / safety)
        let b1 = bankroll * 0.40
        let b2 = bankroll * 0.30
        let b3 = bankroll * 0.30

        // Build parlays/singles using greedy correlation-aware selection
        let p1 = buildParlay(maxLegs: legsPrimary, budget: b1, label: "Primary A")
        let p2 = buildParlay(maxLegs: legsSecondary, budget: b2, label: "Primary B", avoidEventIds: Set(p1.legs.map{ $0.eventId }))
        var p3 = buildParlay(maxLegs: legsSafety, budget: b3, label: "Safety", preferSafer: true, avoidEventIds: Set(p1.legs.map{ $0.eventId }).union(p2.legs.map{ $0.eventId }))

        // If target profit specified and expected combined profit < target, nudge safety stake up if possible
        if target > 0 {
            let expectedProfit = max(0, p1.ev) + max(0, p2.ev) + max(0, p3.ev)
            if expectedProfit < target {
                let extra = min(b3 * 0.5, target - expectedProfit)
                let st = p3.stake + extra
                let m = vm.parlayExpectedValue(stake: st, legs: p3.legs)
                p3 = BetPlan(legs: p3.legs, stake: st, category: p3.category, rationale: p3.rationale + " • boosted to chase target", hitProb: m.hitProb, decimalOdds: m.decimalOdds, payout: m.payout, ev: m.ev)
            }
        }

        plan = [p1, p2, p3]
    }

    private func buildParlay(maxLegs: Int, budget: Double, label: String, preferSafer: Bool = false, avoidEventIds: Set<String> = []) -> BetPlan {
        // Candidate pool filtered by event diversity preference
        var pool = vm.topPicks.filter { !avoidEventIds.contains($0.eventId) }
        guard !pool.isEmpty else {
            return BetPlan(legs: [], stake: 0, category: .Lotto, rationale: "No picks available", hitProb: 0, decimalOdds: 0, payout: 0, ev: 0)
        }

        // Seed with the best single
        pool.sort { $0.confidence > $1.confidence }
        var chosen: [Pick] = []
        var candidates = pool

        let (wConf, wCorr) : (Double, Double) = {
            switch vm.riskMode {
            case .Conservative: return (0.75, 0.25)
            case .Moderate:     return (0.65, 0.35)
            case .Aggressive:   return (0.55, 0.45)
            }
        }()

        while chosen.count < maxLegs, !candidates.isEmpty {
            let scored = candidates.map { cand -> (Pick, Double) in
                let corr = chosen.isEmpty ? 0 : chosen.map { vm.correlationScore($0, cand) }.reduce(0, +) / Double(chosen.count)
                let diversityPenalty = chosen.contains(where: { $0.player == cand.player && $0.marketKey.key == cand.marketKey.key }) ? 0.12 : 0.0
                // If preferSafer, lightly penalize same-game stacks of 3+
                let sameGameCount = chosen.filter{ $0.eventId == cand.eventId }.count
                let sameGamePenalty = (preferSafer && sameGameCount >= 2) ? 0.08 : 0.0
                let composite = wConf * cand.confidence + wCorr * corr - diversityPenalty - sameGamePenalty
                return (cand, composite)
            }
            guard let best = scored.max(by: { $0.1 < $1.1 }) else { break }
            chosen.append(best.0)
            candidates.removeAll { $0.id == best.0.id }
        }

        // Compute metrics & stake sizing
        let stake = max(1, round(budget))
        let m = vm.parlayExpectedValue(stake: stake, legs: chosen)
        let tier: Tier = {
            switch m.hitProb {
            case ..<0.42: return .Lotto
            case 0.42..<0.55: return .Risky
            case 0.55..<0.68: return .Medium
            default: return .Safe
            }
        }()

        let rationale = "\(label): optimized on confidence & correlation (\(chosen.count) leg\(chosen.count == 1 ? "" : "s"))."
        return BetPlan(legs: chosen, stake: stake, category: tier, rationale: rationale, hitProb: m.hitProb, decimalOdds: m.decimalOdds, payout: m.payout, ev: m.ev)
    }
}

// MARK: - Settings (placeholder to restore build)
struct SettingsView: View {
    @ObservedObject var vm: AppVM
    @State private var tmpBankroll: String = ""
    @State private var tmpLossCap: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Bankroll & Risk") {
                    HStack {
                        Text("Bankroll $")
                        TextField("1000", text: Binding(
                            get: { tmpBankroll.isEmpty ? String(Int(vm.bankroll)) : tmpBankroll },
                            set: { tmpBankroll = $0 }
                        ))
                        .keyboardType(.numberPad)
                        .onSubmit {
                            if let b = Double(tmpBankroll) { vm.bankroll = b }
                        }
                    }
                    Picker("Risk Mode", selection: Binding(get: { vm.riskMode }, set: { vm.riskMode = $0 })) {
                        ForEach(RiskMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                }

                Section("Responsible Gaming") {
                    Toggle("Enable Daily Loss Cap", isOn: $vm.dailyLossCapEnabled)
                    if vm.dailyLossCapEnabled {
                        HStack {
                            Text("Loss Cap $")
                            TextField("100", text: Binding(
                                get: { tmpLossCap.isEmpty ? String(Int(vm.dailyLossCapAmount)) : tmpLossCap },
                                set: { tmpLossCap = $0 }
                            ))
                            .keyboardType(.numberPad)
                            .onSubmit {
                                if let c = Double(tmpLossCap) { vm.dailyLossCapAmount = c }
                            }
                        }
                    }
                }

                Section("Modes") {
                    Toggle("Market Parity Mode", isOn: $vm.marketParityMode)
                }
            }
            .navigationTitle("Settings")
            .modifier(BetSlipOverlay(vm: vm))
        }
    }
}

    // Helper for consistent odds display
    private func oddsString(_ odds: Double) -> String {
        odds > 0 ? "+\(Int(odds))" : "\(Int(odds))"
    }
