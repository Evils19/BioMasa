namespace BioMasaWeb.Models;

public class BiomassAnalysisResponse
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DateTime AnalysisDate { get; set; } = DateTime.Now;
    public BiomassComponents Components { get; set; } = new();
    public string ImageBase64 { get; set; } = string.Empty;
    public string Recommendations { get; set; } = string.Empty;
    public double ConfidenceScore { get; set; }
}


public class BiomassComponents
{
    public double DryGreenG { get; set; }
    public double DryCloverG { get; set; }
    public double DryDeadG { get; set; }
    public double DryTotalG { get; set; }
    public double GdmG { get; set; }
}

public class GroqBiomassResponse
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public double DryGreen { get; set; }
    public double DryClover { get; set; }
    public double DryDead { get; set; }
    public double DryTotal { get; set; }
    public double Gdm { get; set; }
    public string Recommendations { get; set; } = string.Empty;
    public double Confidence { get; set; }
}

