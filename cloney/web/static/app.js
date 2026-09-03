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

  // Rückfragen stehen in der Mitte der Seite, nicht am oberen Fensterrand.
  //
  // window.confirm setzt seinen Kasten dorthin, wo der Browser ihn hinsetzt:
  // ganz oben, festgeklebt an der Adressleiste. Wer unten in der Seite auf
  // "Projekt löschen" gedrückt hat, sucht die Antwort dann am anderen Ende des
  // Bildschirms -- und beantwortet im Zweifel eine Frage, die er nicht gelesen
  // hat. Gefragt wird deshalb im Dokument selbst.
  //
  // Fehlt das Element oder kennt der Browser showModal nicht, bleibt
  // window.confirm: eine Rückfrage darf nicht verschwinden, nur weil ihre
  // Verpackung fehlt. Sonst löschte ein Klick kommentarlos ein Projekt.
  function frage(text, jaWort) {
    var kasten = document.getElementById("rueckfrage");
    if (!kasten || !kasten.showModal) return Promise.resolve(window.confirm(text));

    kasten.querySelector("#rueckfrage-text").textContent = text;
    kasten.querySelector("button[value=ja]").textContent = jaWort || "Fortfahren";
    return new Promise(function (fertig) {
      kasten.addEventListener(
        "close",
        function () {
          // Alles außer einem ausdrücklichen Ja ist ein Nein -- Escape und der
          // Klick daneben eingeschlossen.
          fertig(kasten.returnValue === "ja");
        },
        { once: true }
      );
      kasten.returnValue = "";
      kasten.showModal();
    });
  }

  // htmx meldet sich vor jeder Anfrage; ohne hx-confirm ist question null.
  document.body.addEventListener("htmx:confirm", function (ereignis) {
    var text = ereignis.detail.question;
    if (!text) return;
    ereignis.preventDefault();
    var wort = ereignis.detail.elt.getAttribute("data-frage-ja");
    frage(text, wort).then(function (ja) {
      if (ja) ereignis.detail.issueRequest(true);
    });
  });

  // Formulare ohne htmx. Die Frage hängt am Knopf, der sie auslöst, nicht am
  // Formular: auf der Stimmenseite sitzen "speichern" und "löschen" im selben.
  //
  // Gehorcht wird in der Fangphase, also bevor irgendein anderer Zuhörer den
  // Klick sieht. Nach einem Ja wird derselbe Knopf noch einmal gedrückt, diesmal
  // mit Vermerk -- so bleibt formaction, formnovalidate und alles Weitere am
  // Knopf gültig, statt hier nachgebaut zu werden.
  document.addEventListener(
    "click",
    function (ereignis) {
      var ziel = ereignis.target;
      var knopf = ziel && ziel.closest ? ziel.closest("[data-frage]") : null;
      if (!knopf || knopf.bestaetigt) return;
      ereignis.preventDefault();
      ereignis.stopPropagation();
      frage(knopf.dataset.frage, knopf.dataset.frageJa).then(function (ja) {
        if (!ja) return;
        knopf.bestaetigt = true;
        knopf.click();
        knopf.bestaetigt = false;
      });
    },
    true
  );

  // Ein Seitenfenster: derselbe dialog, nur an den Rand gestellt.
  //
  // Die Textprobe eines Vergleichs ist das Längste am Formular und das, was man
  // am seltensten anfasst. Ausgeklappt schob sie alles Übrige unter den
  // Bildschirmrand -- die Achsen, die man tatsächlich einstellt, standen dann
  // außer Sicht.
  //
  // Das Feld bleibt Teil des Formulars, auch solange das Fenster zu ist: es
  // liegt im Formular, und ein geschlossener dialog nimmt nichts aus dem
  // Absenden heraus.
  document.addEventListener("click", function (ereignis) {
    var ziel = ereignis.target;
    if (!ziel || !ziel.closest) return;

    var oeffner = ziel.closest("[data-oeffnet]");
    if (oeffner) {
      var fenster = document.querySelector(oeffner.dataset.oeffnet);
      if (fenster && fenster.showModal) {
        ereignis.preventDefault();
        fenster.showModal();
        var feld = fenster.querySelector("textarea, input");
        if (feld) feld.focus();
      }
      return;
    }

    var schliesser = ziel.closest("[data-schliesst]");
    if (schliesser) {
      var offen = schliesser.closest("dialog");
      if (offen) {
        ereignis.preventDefault();
        offen.close();
      }
      return;
    }

    // Einen Wert aus einer Achse nehmen. Danach ausdrücklich ein change
    // auslösen: die Vorschau hängt daran, und das Entfernen geschieht hier
    // statt über eine Anfrage.
    var entferner = ziel.closest("[data-entfernen]");
    if (entferner) {
      ereignis.preventDefault();
      var feldgruppe = entferner.closest(".wertfeld");
      var behaelter = feldgruppe && feldgruppe.parentElement;
      // Der letzte Wert einer Achse bleibt stehen: eine Achse ganz ohne Wert
      // wäre kein leeres Raster, sondern ein verschwundener Regler.
      if (behaelter && behaelter.querySelectorAll(".wertfeld").length > 1) {
        feldgruppe.remove();
        behaelter.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
  });

  // Die Zeichenzahl am Knopf gilt sonst nur bis zur ersten Eingabe.
  document.addEventListener("close", function (ereignis) {
    var fenster = ereignis.target;
    if (!fenster || !fenster.querySelector) return;
    var feld = fenster.querySelector("textarea");
    var zaehler = document.querySelector("[data-zeichenzahl]");
    if (!feld || !zaehler) return;
    var laenge = feld.value.length;
    zaehler.textContent = laenge ? laenge + " Zeichen" : "noch leer";
  }, true);

  document.body.addEventListener("htmx:responseError", function (ereignis) {
    melden(fehlertext(ereignis.detail.xhr), "error");
  });

  document.body.addEventListener("htmx:sendError", function () {
    melden("Keine Verbindung zum Server. Läuft 'cloney web' noch?", "error");
  });

  // Während eines Laufs holt sich die Satztabelle alle zwei Sekunden den neuen
  // Stand. Wer dabei einen fertigen Satz anhört, verlöre ihn bei jedem Tausch:
  // das Element wird ersetzt, der Ton beginnt von vorn zu laden.
  //
  // Versucht wurde zuerst, Zeitpunkt und Wiedergabe danach wiederherzustellen.
  // Im Browser gemessen hat das zwar funktioniert, aber mit einem hörbaren
  // Aussetzer je Tausch -- alle zwei Sekunden. Deshalb gilt hier die einfachere
  // Regel: Zuhören geht vor. Solange etwas läuft, wird die wiederkehrende
  // Abfrage übersprungen; danach holt die nächste alles nach.
  //
  // Betroffen sind nur wiederkehrende Abfragen. Ein Klick auf "Neu würfeln"
  // wird nie zurückgestellt.
  function spieltEtwas(bereich) {
    var spieler = bereich.querySelectorAll ? bereich.querySelectorAll("audio") : [];
    for (var i = 0; i < spieler.length; i++) {
      if (!spieler[i].paused && !spieler[i].ended) return true;
    }
    return false;
  }

  document.body.addEventListener("htmx:beforeRequest", function (ereignis) {
    var quelle = ereignis.detail.elt;
    if (!quelle || !quelle.getAttribute) return;
    var ausloeser = quelle.getAttribute("hx-trigger") || "";
    if (ausloeser.indexOf("every") === -1) return;
    if (spieltEtwas(quelle)) ereignis.preventDefault();
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
