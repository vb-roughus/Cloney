// Rückmeldung für alles, was länger dauert als ein Klick.
//
// htmx tauscht bei Antworten mit Fehlerstatus nichts ein und meldet von sich aus
// auch nichts. Ein abgelehnter Klick sieht damit genauso aus wie ein wirkungsloser
// -- deshalb werden Fehler hier ausdrücklich sichtbar gemacht.

(function () {
  "use strict";

  function melden(text, art) {
    var behaelter = document.getElementById("meldungen");
    if (!behaelter) return;
    var eintrag = document.createElement("div");
    eintrag.className = "toast " + (art || "error");
    eintrag.setAttribute("role", "status");
    eintrag.textContent = text;
    var schliessen = document.createElement("button");
    schliessen.className = "toast-close";
    schliessen.setAttribute("aria-label", "Meldung schließen");
    schliessen.textContent = "×";
    schliessen.addEventListener("click", function () {
      eintrag.remove();
    });
    eintrag.appendChild(schliessen);
    behaelter.appendChild(eintrag);
    if (art === "info") {
      setTimeout(function () {
        eintrag.remove();
      }, 6000);
    }
  }

  // FastAPI verpackt HTTPException als {"detail": "..."}. Alles andere wird
  // roh gezeigt, statt es zu verschlucken.
  function fehlertext(antwort) {
    var rohtext = antwort.responseText || "";
    try {
      var daten = JSON.parse(rohtext);
      if (daten && typeof daten.detail === "string") return daten.detail;
    } catch (e) {
      /* kein JSON -- dann eben der Rohtext */
    }
    if (rohtext && rohtext.length < 300) return rohtext;
    return "Der Server antwortete mit Status " + antwort.status + ".";
  }

  document.body.addEventListener("htmx:responseError", function (ereignis) {
    melden(fehlertext(ereignis.detail.xhr), "error");
  });

  document.body.addEventListener("htmx:sendError", function () {
    melden("Keine Verbindung zum Server. Läuft 'cloney web' noch?", "error");
  });

  // Ein laufender Renderlauf soll nicht unbemerkt fertig werden. Gemeldet wird
  // der Übergang von "läuft" nach "fertig", nicht der Zustand selbst -- sonst
  // käme die Meldung bei jedem Seitenaufruf erneut.
  //
  // Gelesen wird dabei das Dokument, nicht das Ereignis: bei einem
  // outerHTML-Tausch verweist detail.target auf das ersetzte, also alte
  // Element. Dessen Zustand ist genau der von vor dem Tausch.
  var liefBisher = false;
  document.body.addEventListener("htmx:afterSettle", function () {
    var karte = document.getElementById("status");
    if (!karte || !karte.dataset.fertig) return;
    var laeuft = karte.dataset.fertig !== "1";
    if (liefBisher && !laeuft) melden("Renderlauf abgeschlossen.", "info");
    liefBisher = laeuft;
  });
})();
