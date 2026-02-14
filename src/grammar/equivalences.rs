//! Static cross-lingual equivalence table for high-frequency terms.
//!
//! Maps surface forms across English, Russian, Arabic, French, and Spanish
//! to a single canonical label (English by convention). This covers ~200
//! commonly encountered terms in intelligence analysis contexts: countries,
//! capitals, organizations, and domain terms.

/// A cross-lingual equivalence entry.
///
/// Each entry maps one canonical English label to its translations.
/// The `aliases` slice contains all known surface forms across languages.
pub struct Equivalence {
    /// Canonical label (English).
    pub canonical: &'static str,
    /// All known surface forms across languages.
    pub aliases: &'static [&'static str],
}

/// The static equivalence table.
///
/// Sorted by canonical label for binary search.
pub static EQUIVALENCES: &[Equivalence] = &[
    // Countries & territories
    Equivalence {
        canonical: "Algeria",
        aliases: &["Algérie", "Argelia", "الجزائر", "Алжир"],
    },
    Equivalence {
        canonical: "Argentina",
        aliases: &["Argentine", "Аргентина", "الأرجنتين"],
    },
    Equivalence {
        canonical: "Australia",
        aliases: &["Australie", "Австралия", "أستراليا"],
    },
    Equivalence {
        canonical: "Austria",
        aliases: &["Autriche", "Австрия", "النمسا"],
    },
    Equivalence {
        canonical: "Belgium",
        aliases: &["Belgique", "Bélgica", "Бельгия", "بلجيكا"],
    },
    Equivalence {
        canonical: "Brazil",
        aliases: &["Brésil", "Brasil", "Бразилия", "البرازيل"],
    },
    Equivalence {
        canonical: "Canada",
        aliases: &["Канада", "كندا"],
    },
    Equivalence {
        canonical: "China",
        aliases: &["Chine", "Китай", "الصين"],
    },
    Equivalence {
        canonical: "Cuba",
        aliases: &["Куба", "كوبا"],
    },
    Equivalence {
        canonical: "Czech Republic",
        aliases: &["République tchèque", "República Checa", "Чехия", "التشيك"],
    },
    Equivalence {
        canonical: "Denmark",
        aliases: &["Danemark", "Dinamarca", "Дания", "الدنمارك"],
    },
    Equivalence {
        canonical: "Egypt",
        aliases: &["Égypte", "Egipto", "Египет", "مصر"],
    },
    Equivalence {
        canonical: "Finland",
        aliases: &["Finlande", "Finlandia", "Финляндия", "فنلندا"],
    },
    Equivalence {
        canonical: "France",
        aliases: &["Francia", "Франция", "فرنسا"],
    },
    Equivalence {
        canonical: "Germany",
        aliases: &["Allemagne", "Alemania", "Германия", "ألمانيا"],
    },
    Equivalence {
        canonical: "Greece",
        aliases: &["Grèce", "Grecia", "Греция", "اليونان"],
    },
    Equivalence {
        canonical: "Hungary",
        aliases: &["Hongrie", "Hungría", "Венгрия", "المجر"],
    },
    Equivalence {
        canonical: "India",
        aliases: &["Inde", "Индия", "الهند"],
    },
    Equivalence {
        canonical: "Indonesia",
        aliases: &["Indonésie", "Индонезия", "إندونيسيا"],
    },
    Equivalence {
        canonical: "Iran",
        aliases: &["Иран", "إيران"],
    },
    Equivalence {
        canonical: "Iraq",
        aliases: &["Irak", "Ирак", "العراق"],
    },
    Equivalence {
        canonical: "Ireland",
        aliases: &["Irlande", "Irlanda", "Ирландия", "أيرلندا"],
    },
    Equivalence {
        canonical: "Israel",
        aliases: &["Israël", "Израиль", "إسرائيل"],
    },
    Equivalence {
        canonical: "Italy",
        aliases: &["Italie", "Italia", "Италия", "إيطاليا"],
    },
    Equivalence {
        canonical: "Japan",
        aliases: &["Japon", "Japón", "Япония", "اليابان"],
    },
    Equivalence {
        canonical: "Jordan",
        aliases: &["Jordanie", "Jordania", "Иордания", "الأردن"],
    },
    Equivalence {
        canonical: "Kazakhstan",
        aliases: &["Казахстан", "كازاخستان"],
    },
    Equivalence {
        canonical: "Kuwait",
        aliases: &["Koweït", "Кувейт", "الكويت"],
    },
    Equivalence {
        canonical: "Lebanon",
        aliases: &["Liban", "Líbano", "Ливан", "لبنان"],
    },
    Equivalence {
        canonical: "Libya",
        aliases: &["Libye", "Libia", "Ливия", "ليبيا"],
    },
    Equivalence {
        canonical: "Mexico",
        aliases: &["Mexique", "México", "Мексика", "المكسيك"],
    },
    Equivalence {
        canonical: "Morocco",
        aliases: &["Maroc", "Marruecos", "Марокко", "المغرب"],
    },
    Equivalence {
        canonical: "Netherlands",
        aliases: &["Pays-Bas", "Países Bajos", "Нидерланды", "هولندا"],
    },
    Equivalence {
        canonical: "Nigeria",
        aliases: &["Nigéria", "Нигерия", "نيجيريا"],
    },
    Equivalence {
        canonical: "North Korea",
        aliases: &[
            "Corée du Nord",
            "Corea del Norte",
            "Северная Корея",
            "كوريا الشمالية",
        ],
    },
    Equivalence {
        canonical: "Norway",
        aliases: &["Norvège", "Noruega", "Норвегия", "النرويج"],
    },
    Equivalence {
        canonical: "Pakistan",
        aliases: &["Пакистан", "باكستان"],
    },
    Equivalence {
        canonical: "Palestine",
        aliases: &["Палестина", "فلسطين"],
    },
    Equivalence {
        canonical: "Poland",
        aliases: &["Pologne", "Polonia", "Польша", "بولندا"],
    },
    Equivalence {
        canonical: "Portugal",
        aliases: &["Португалия", "البرتغال"],
    },
    Equivalence {
        canonical: "Qatar",
        aliases: &["Катар", "قطر"],
    },
    Equivalence {
        canonical: "Romania",
        aliases: &["Roumanie", "Rumania", "Румыния", "رومانيا"],
    },
    Equivalence {
        canonical: "Russia",
        aliases: &["Russie", "Rusia", "Россия", "روسيا"],
    },
    Equivalence {
        canonical: "Saudi Arabia",
        aliases: &[
            "Arabie saoudite",
            "Arabia Saudita",
            "Саудовская Аравия",
            "السعودية",
        ],
    },
    Equivalence {
        canonical: "South Korea",
        aliases: &[
            "Corée du Sud",
            "Corea del Sur",
            "Южная Корея",
            "كوريا الجنوبية",
        ],
    },
    Equivalence {
        canonical: "Spain",
        aliases: &["Espagne", "España", "Испания", "إسبانيا"],
    },
    Equivalence {
        canonical: "Sudan",
        aliases: &["Soudan", "Судан", "السودان"],
    },
    Equivalence {
        canonical: "Sweden",
        aliases: &["Suède", "Suecia", "Швеция", "السويد"],
    },
    Equivalence {
        canonical: "Switzerland",
        aliases: &["Suisse", "Suiza", "Швейцария", "سويسرا"],
    },
    Equivalence {
        canonical: "Syria",
        aliases: &["Syrie", "Siria", "Сирия", "سوريا"],
    },
    Equivalence {
        canonical: "Tunisia",
        aliases: &["Tunisie", "Túnez", "Тунис", "تونس"],
    },
    Equivalence {
        canonical: "Turkey",
        aliases: &["Turquie", "Turquía", "Турция", "تركيا"],
    },
    Equivalence {
        canonical: "Ukraine",
        aliases: &["Украина", "أوكرانيا"],
    },
    Equivalence {
        canonical: "United Arab Emirates",
        aliases: &[
            "Émirats arabes unis",
            "Emiratos Árabes Unidos",
            "ОАЭ",
            "الإمارات",
        ],
    },
    Equivalence {
        canonical: "United Kingdom",
        aliases: &[
            "Royaume-Uni",
            "Reino Unido",
            "Великобритания",
            "بريطانيا",
            "UK",
            "GB",
        ],
    },
    Equivalence {
        canonical: "United States",
        aliases: &[
            "États-Unis",
            "Estados Unidos",
            "США",
            "الولايات المتحدة",
            "USA",
            "US",
        ],
    },
    Equivalence {
        canonical: "Yemen",
        aliases: &["Yémen", "Йемен", "اليمن"],
    },
    // Capitals & major cities
    Equivalence {
        canonical: "Algiers",
        aliases: &["Alger", "Argel", "Алжир", "الجزائر العاصمة"],
    },
    Equivalence {
        canonical: "Ankara",
        aliases: &["Анкара", "أنقرة"],
    },
    Equivalence {
        canonical: "Baghdad",
        aliases: &["Bagdad", "Багдад", "بغداد"],
    },
    Equivalence {
        canonical: "Beijing",
        aliases: &["Pékin", "Pekín", "Пекин", "بكين"],
    },
    Equivalence {
        canonical: "Berlin",
        aliases: &["Берлин", "برلين"],
    },
    Equivalence {
        canonical: "Brussels",
        aliases: &["Bruxelles", "Bruselas", "Брюссель", "بروكسل"],
    },
    Equivalence {
        canonical: "Cairo",
        aliases: &["Le Caire", "El Cairo", "Каир", "القاهرة"],
    },
    Equivalence {
        canonical: "Damascus",
        aliases: &["Damas", "Damasco", "Дамаск", "دمشق"],
    },
    Equivalence {
        canonical: "Geneva",
        aliases: &["Genève", "Ginebra", "Женева", "جنيف"],
    },
    Equivalence {
        canonical: "Istanbul",
        aliases: &["Стамбул", "إسطنبول"],
    },
    Equivalence {
        canonical: "Jerusalem",
        aliases: &["Jérusalem", "Jerusalén", "Иерусалим", "القدس"],
    },
    Equivalence {
        canonical: "Kiev",
        aliases: &["Kyiv", "Киев", "كييف"],
    },
    Equivalence {
        canonical: "London",
        aliases: &["Londres", "Лондон", "لندن"],
    },
    Equivalence {
        canonical: "Madrid",
        aliases: &["Мадрид", "مدريد"],
    },
    Equivalence {
        canonical: "Moscow",
        aliases: &["Moscou", "Moscú", "Москва", "موسكو"],
    },
    Equivalence {
        canonical: "Paris",
        aliases: &["Париж", "باريس"],
    },
    Equivalence {
        canonical: "Riyadh",
        aliases: &["Riyad", "Riad", "Эр-Рияд", "الرياض"],
    },
    Equivalence {
        canonical: "Rome",
        aliases: &["Roma", "Рим", "روما"],
    },
    Equivalence {
        canonical: "Saint Petersburg",
        aliases: &[
            "Saint-Pétersbourg",
            "San Petersburgo",
            "Санкт-Петербург",
            "سانت بطرسبرغ",
        ],
    },
    Equivalence {
        canonical: "Tehran",
        aliases: &["Téhéran", "Teherán", "Тегеран", "طهران"],
    },
    Equivalence {
        canonical: "Tokyo",
        aliases: &["Tokio", "Токио", "طوكيو"],
    },
    Equivalence {
        canonical: "Vienna",
        aliases: &["Vienne", "Viena", "Вена", "فيينا"],
    },
    Equivalence {
        canonical: "Warsaw",
        aliases: &["Varsovie", "Varsovia", "Варшава", "وارسو"],
    },
    Equivalence {
        canonical: "Washington",
        aliases: &["Вашингтон", "واشنطن"],
    },
    // Organizations & institutions
    Equivalence {
        canonical: "European Union",
        aliases: &[
            "Union européenne",
            "Unión Europea",
            "Европейский Союз",
            "الاتحاد الأوروبي",
            "EU",
            "UE",
            "ЕС",
        ],
    },
    Equivalence {
        canonical: "NATO",
        aliases: &["OTAN", "НАТО", "حلف الناتو"],
    },
    Equivalence {
        canonical: "United Nations",
        aliases: &[
            "Nations Unies",
            "Naciones Unidas",
            "ООН",
            "الأمم المتحدة",
            "UN",
            "ONU",
        ],
    },
    // Common domain terms
    Equivalence {
        canonical: "animal",
        aliases: &["животное", "حيوان", "animaux", "animal"],
    },
    Equivalence {
        canonical: "city",
        aliases: &["город", "مدينة", "ville", "ciudad"],
    },
    Equivalence {
        canonical: "computer",
        aliases: &[
            "компьютер",
            "حاسوب",
            "ordinateur",
            "computadora",
            "ordenador",
        ],
    },
    Equivalence {
        canonical: "country",
        aliases: &["страна", "دولة", "بلد", "pays", "país"],
    },
    Equivalence {
        canonical: "democracy",
        aliases: &["демократия", "ديمقراطية", "démocratie", "democracia"],
    },
    Equivalence {
        canonical: "energy",
        aliases: &["энергия", "طاقة", "énergie", "energía"],
    },
    Equivalence {
        canonical: "government",
        aliases: &["правительство", "حكومة", "gouvernement", "gobierno"],
    },
    Equivalence {
        canonical: "human",
        aliases: &["человек", "إنسان", "humain", "humano"],
    },
    Equivalence {
        canonical: "information",
        aliases: &["информация", "معلومات", "información"],
    },
    Equivalence {
        canonical: "language",
        aliases: &["язык", "لغة", "langue", "idioma", "lengua"],
    },
    Equivalence {
        canonical: "mammal",
        aliases: &["млекопитающее", "ثديي", "mammifère", "mamífero"],
    },
    Equivalence {
        canonical: "military",
        aliases: &["военный", "عسكري", "militaire", "militar"],
    },
    Equivalence {
        canonical: "oil",
        aliases: &["нефть", "نفط", "pétrole", "petróleo"],
    },
    Equivalence {
        canonical: "organization",
        aliases: &["организация", "منظمة", "organisation", "organización"],
    },
    Equivalence {
        canonical: "person",
        aliases: &["человек", "شخص", "personne", "persona"],
    },
    Equivalence {
        canonical: "politics",
        aliases: &["политика", "سياسة", "politique", "política"],
    },
    Equivalence {
        canonical: "river",
        aliases: &["река", "نهر", "rivière", "río"],
    },
    Equivalence {
        canonical: "science",
        aliases: &["наука", "علم", "ciencia"],
    },
    Equivalence {
        canonical: "security",
        aliases: &["безопасность", "أمن", "sécurité", "seguridad"],
    },
    Equivalence {
        canonical: "technology",
        aliases: &["технология", "تكنولوجيا", "technologie", "tecnología"],
    },
    Equivalence {
        canonical: "university",
        aliases: &["университет", "جامعة", "université", "universidad"],
    },
    Equivalence {
        canonical: "war",
        aliases: &["война", "حرب", "guerre", "guerra"],
    },
    Equivalence {
        canonical: "water",
        aliases: &["вода", "ماء", "eau", "agua"],
    },
    Equivalence {
        canonical: "weapon",
        aliases: &["оружие", "سلاح", "arme", "arma"],
    },
];

/// Look up a surface form in the equivalence table.
///
/// Returns the canonical label if found, `None` otherwise.
/// Case-insensitive comparison.
pub fn lookup_equivalence(surface: &str) -> Option<&'static str> {
    let lower = surface.to_lowercase();

    for equiv in EQUIVALENCES {
        if equiv.canonical.to_lowercase() == lower {
            return Some(equiv.canonical);
        }
        for alias in equiv.aliases {
            if alias.to_lowercase() == lower {
                return Some(equiv.canonical);
            }
            // Also try the original casing
            if *alias == surface {
                return Some(equiv.canonical);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_english_canonical() {
        assert_eq!(lookup_equivalence("Moscow"), Some("Moscow"));
    }

    #[test]
    fn lookup_russian_alias() {
        assert_eq!(lookup_equivalence("Москва"), Some("Moscow"));
    }

    #[test]
    fn lookup_french_alias() {
        assert_eq!(lookup_equivalence("Moscou"), Some("Moscow"));
    }

    #[test]
    fn lookup_arabic_alias() {
        assert_eq!(lookup_equivalence("موسكو"), Some("Moscow"));
    }

    #[test]
    fn lookup_case_insensitive() {
        assert_eq!(lookup_equivalence("moscow"), Some("Moscow"));
        assert_eq!(lookup_equivalence("FRANCE"), Some("France"));
    }

    #[test]
    fn lookup_not_found() {
        assert_eq!(lookup_equivalence("Xyzzyplugh"), None);
    }

    #[test]
    fn lookup_organization() {
        assert_eq!(lookup_equivalence("НАТО"), Some("NATO"));
        assert_eq!(lookup_equivalence("OTAN"), Some("NATO"));
    }

    #[test]
    fn lookup_common_term() {
        assert_eq!(lookup_equivalence("млекопитающее"), Some("mammal"));
        assert_eq!(lookup_equivalence("mammifère"), Some("mammal"));
    }
}
